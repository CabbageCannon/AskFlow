import "./SuggestionCards.css";

type SuggestionCardsProps = {
  disabled: boolean;
  onSelect: (prompt: string) => void;
};

const suggestions = [
  {
    label: "学习",
    prompt: "帮我制定一个循序渐进的前端学习计划",
  },
  {
    label: "代码",
    prompt: "解释 React 中 useEffect 的常见使用场景和注意事项",
  },
  {
    label: "创意",
    prompt: "为一个 AI 对话应用提供三个实用的功能创意",
  },
  {
    label: "总结",
    prompt: "告诉我如何快速总结一篇技术文章的核心内容",
  },
] as const;

const SuggestionCards = ({ disabled, onSelect }: SuggestionCardsProps) => {
  return (
    <section className="suggestionSection" aria-labelledby="suggestion-title">
      <div className="suggestionHeading">
        <h1 id="suggestion-title">今天想聊点什么？</h1>
        <p>选择一个方向开始，或者直接在下方输入问题。</p>
      </div>

      <div className="suggestionGrid">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion.prompt}
            type="button"
            className="suggestionCard"
            disabled={disabled}
            onClick={() => onSelect(suggestion.prompt)}
          >
            <span className="suggestionLabel">{suggestion.label}</span>
            <span className="suggestionPrompt">{suggestion.prompt}</span>
          </button>
        ))}
      </div>
    </section>
  );
};

export default SuggestionCards;
