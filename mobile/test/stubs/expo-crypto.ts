export function getRandomBytes(count: number): Uint8Array {
  const bytes = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    bytes[index] = (index * 37 + 11) % 256;
  }
  return bytes;
}
