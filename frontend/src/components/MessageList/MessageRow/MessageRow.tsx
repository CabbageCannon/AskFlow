import "./MessageRow.css"
import MarkdownRender from "@/components/MarkdownRender/MarkdownRender";
import type { Message } from "@/type/chat";

type MessageRowProps = {
  message: Message;
  isGenerating: boolean;
  onRegenerate: (messageId: string) => Promise<void>;
};

const MessageRow = ({
  message,
  isGenerating,
  onRegenerate,
}: MessageRowProps) => {
  const canRegenerate = message.role === "assistant"
    && message.status !== "generating"
    && message.status !== "pending";

  return <div
    key={message.id}
    className={`message-item ${message.role}`}>
    <div className="message-main">
      <div className="avatar">
        {message.role === "user" ? "我" : "AI"}
      </div>
      <div className="message-content">
        {/* agent运行过程label */}
        {
          (message.status === "generating"
            && message.content === "") ?
            <div className="research=-progress">
              <span className="research-dot">

                {message.researchProgress.label}
              </span>
            </div> : null}
        {
          message.content ?
            (<MarkdownRender content={message.content} />) : null

        }

        {message.status === "failed" && message.error ? (
          <div className="message-error">{"错误"+message.error}</div>
        ) : null}

        {canRegenerate ? (
          <div className="message-actions">
            <button
              type="button"
              className="regenerate-button"
              disabled={isGenerating}
              onClick={() => {
                void onRegenerate(message.id);
              }}
            >
              重新生成
            </button>
          </div>
        ) : null}
      </div>
    </div>
  </div>
}

export default MessageRow;
