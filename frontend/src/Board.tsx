import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Key } from "chessground/types";

interface Props {
  fen: string;
  bestMove: string | null; // UCI, drawn as an arrow
}

export default function Board({ fen, bestMove }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const api = useRef<Api | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const shapes = bestMove
      ? [
          {
            orig: bestMove.slice(0, 2) as Key,
            dest: bestMove.slice(2, 4) as Key,
            brush: "green",
          },
        ]
      : [];
    if (api.current) {
      api.current.set({ fen, drawable: { autoShapes: shapes } });
    } else {
      api.current = Chessground(ref.current, {
        fen,
        viewOnly: true,
        coordinates: false,
        drawable: { autoShapes: shapes },
      });
    }
  }, [fen, bestMove]);

  useEffect(() => () => api.current?.destroy(), []);

  return <div className="board" ref={ref} />;
}
