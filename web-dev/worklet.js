// Development host: runs the REAL component.wasm — the same bytes RackForge
// loads — inside an AudioWorklet, speaking the wasm-v1 ABI directly. Nothing
// is reimplemented: if it sounds wrong here, it is wrong in the plugin.
//
// The wasm bytes arrive via processorOptions (fetch() does not exist in an
// AudioWorkletGlobalScope), and compilation is synchronous — the module is
// ~18 KB, far below any audible hiccup at construction time.

class RfDrumsProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const module = new WebAssembly.Module(options.processorOptions.wasmBytes);
    const instance = new WebAssembly.Instance(module, {});
    this.abi = instance.exports;
    this.memory = this.abi.memory;

    const status = this.abi.rackforge_prepare(sampleRate, 128, 0, 2);
    if (status !== 0) {
      throw new Error(`rackforge_prepare failed: ${status}`);
    }

    this.outputPtr = this.abi.rackforge_output_ptr();
    this.midiPtr = this.abi.rackforge_midi_ptr();
    this.midiCapacity = this.abi.rackforge_capacity_midi_events();
    this.pendingMidi = [];

    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "midi") {
        // [status, data1, data2] — struck at the head of the next block.
        this.pendingMidi.push(message.data);
      } else if (message.type === "param") {
        this.abi.rackforge_set_parameter(message.index, message.value);
      } else if (message.type === "reset") {
        this.abi.rackforge_reset();
      } else if (message.type === "read") {
        // Read a run of parameters back out of the engine — the fader
        // panel seeds itself from the wasm's own defaults, so the table in
        // Rust stays the single source of truth.
        const values = message.indices.map((index) =>
          this.abi.rackforge_get_parameter(index),
        );
        this.port.postMessage({ type: "values", tag: message.tag, values });
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const frames = output[0].length; // 128 by spec

    let midiCount = 0;
    if (this.pendingMidi.length > 0) {
      // The ABI packs one event per u64: frame | d0<<32 | d1<<40 | d2<<48
      // | length<<56, mirroring MidiEvent::from_packed.
      const events = new BigUint64Array(
        this.memory.buffer,
        this.midiPtr,
        this.midiCapacity,
      );
      const take = Math.min(this.pendingMidi.length, this.midiCapacity);
      for (let i = 0; i < take; i++) {
        const [d0, d1, d2] = this.pendingMidi[i];
        events[i] =
          (BigInt(d0) << 32n) |
          (BigInt(d1) << 40n) |
          (BigInt(d2) << 48n) |
          (3n << 56n);
      }
      this.pendingMidi.length = 0;
      midiCount = take;
    }

    const status = this.abi.rackforge_process(frames, 0, 2, midiCount, 0);
    if (status !== 0) {
      output[0].fill(0);
      if (output[1]) output[1].fill(0);
      return true;
    }

    // The component writes interleaved frames×channels; the Web Audio graph
    // wants planar channels. (A new Float32Array view every block, because
    // memory.buffer is detached whenever the wasm memory grows.)
    const interleaved = new Float32Array(
      this.memory.buffer,
      this.outputPtr,
      frames * 2,
    );
    const left = output[0];
    const right = output[1] ?? output[0];
    for (let frame = 0; frame < frames; frame++) {
      left[frame] = interleaved[frame * 2];
      right[frame] = interleaved[frame * 2 + 1];
    }
    return true;
  }
}

registerProcessor("rf-drums", RfDrumsProcessor);
