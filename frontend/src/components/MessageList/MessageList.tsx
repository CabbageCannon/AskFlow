import { useChatStore } from "@/store/chatStore";
import { Virtuoso } from "react-virtuoso";
import MessageRow from "./MessageRow/MessageRow";
import "./MessageList.css";

type MessageListProps = {
  isGenerating: boolean;
  onRegenerate: (messageId: string) => Promise<void>;
}

const MessageList = ({ isGenerating, onRegenerate }: MessageListProps) => {
  const currentSessionId = useChatStore(state => state.currentSession);
  const sessions = useChatStore(state => state.sessions);
  const currentMessages = sessions.find((session) => session.id === currentSessionId)?.messages ?? [];
  const isAtBottom = useChatStore(state => state.isAtBottom);
  const setIsAtBottom = useChatStore(state => state.setIsAtBottom);

  return (<div className="message-list">
    <Virtuoso
      className="chat-virtuoso"
      data={currentMessages}
      computeItemKey={(_, message) => message.id}
      itemContent={(_, message) => {
        return <MessageRow
          message={message}
          isGenerating={isGenerating}
          onRegenerate={onRegenerate}
        />
      }}
      followOutput={isAtBottom ? "auto" : false}
      atBottomStateChange={(bottom) => {
        setIsAtBottom(bottom)
      }}
      overscan={50}
    />
  </div>)
}

export default MessageList;
