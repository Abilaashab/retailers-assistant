// Type declarations for TextDecoder
declare class TextDecoder {
  constructor(encoding?: string, options?: { fatal?: boolean; ignoreBOM?: boolean });
  decode(input?: BufferSource, options?: { stream?: boolean }): string;
  readonly encoding: string;
  readonly fatal: boolean;
  readonly ignoreBOM: boolean;
}

declare const TextDecoder: {
  prototype: TextDecoder;
  new(encoding?: string, options?: { fatal?: boolean; ignoreBOM?: boolean }): TextDecoder;
};
