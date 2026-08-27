import { useChatStore } from "@/store/chatStore";
import { useSendMessage } from "@/hooks/useSendMessage";
import MessageList from "../MessageList/MessageList";
import SearchBox from "../SearchBox/SearchBox";
import SuggestionCards from "../SuggestionCards/SuggestionCards";
import "./Main.css";
import { useEffect } from "react";

const Main = () => {
  const ensureCurrentSession = useChatStore(state => state.ensureCurrentSession);

  useEffect(() => {
    ensureCurrentSession();
  }, [ensureCurrentSession]);
  const hasMessages = useChatStore((state) => {
    const currentSession = state.sessions.find(
      (session) => session.id === state.currentSession,
    );

    return Boolean(currentSession?.messages.length);
  });

  const {
    isGenerating,
    regenerateMessage,
    sendMessage,
    stopMessage,
  } = useSendMessage();

  return (
    <div className="main">
      <div className="main-top">
        {hasMessages ? (
          <MessageList
            isGenerating={isGenerating}
            onRegenerate={regenerateMessage}
          />
        ) : (
          <SuggestionCards
            disabled={isGenerating}
            onSelect={(prompt) => {
              void sendMessage(prompt)
            }}
          />
        )}
      </div>

      <div className="main-bottom">
        <SearchBox
          isGenerating={isGenerating}
          sendMessage={sendMessage}
          stopMessage={stopMessage}
        />
      </div>
    </div>
  );
};

export default Main;
