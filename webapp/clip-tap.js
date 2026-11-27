// Audio worklet of the home page's "Match a part" tab capture (search.js):
// hands the sound that reaches it to the page as mono samples, block by
// block. It plays nothing and keeps nothing.
registerProcessor(
  "clip-tap",
  class extends AudioWorkletProcessor {
    process(inputs) {
      const channels = inputs[0];
      if (channels && channels.length && channels[0].length) {
        const mono = new Float32Array(channels[0].length);
        for (const channel of channels) {
          for (let index = 0; index < mono.length; index += 1) {
            mono[index] += channel[index] / channels.length;
          }
        }
        this.port.postMessage(mono, [mono.buffer]);
      }
      return true;
    }
  }
);
