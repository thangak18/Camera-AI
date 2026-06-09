import { Search, Sparkles } from "lucide-react";
import { FormEvent, useState } from "react";

type Props = {
  loading: boolean;
  onSearch: (query: string) => void;
};

const SUGGESTIONS = [
  "Tìm người áo đen đi vào cổng chính",
  "Tìm xe máy màu đỏ ở bãi xe",
  "Tóm tắt sự kiện đáng chú ý hôm nay"
];

export function SearchPanel({ loading, onSearch }: Props) {
  const [query, setQuery] = useState(SUGGESTIONS[0]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSearch(query);
  }

  return (
    <section className="panel search-panel">
      <div className="panel-title">
        <Search size={18} />
        <h2>Truy vấn</h2>
      </div>
      <form onSubmit={submit}>
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          rows={4}
          placeholder="Nhập câu hỏi tiếng Việt..."
        />
        <button className="primary-button" disabled={loading || !query.trim()} type="submit">
          <Sparkles size={18} />
          {loading ? "Đang tìm" : "Tìm bằng AI"}
        </button>
      </form>
      <div className="chips">
        {SUGGESTIONS.map((item) => (
          <button key={item} type="button" onClick={() => setQuery(item)}>
            {item}
          </button>
        ))}
      </div>
    </section>
  );
}

