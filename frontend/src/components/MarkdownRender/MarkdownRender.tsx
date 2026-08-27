import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import "./MarkdownRender.css"
import Icons from "@/assets/icons/icons"
import React from "react";
import type { MouseEvent } from "react";

type Props = {
  content: string
}

const MarkdownRender = ({ content }: Props) => {
  const handleCopy = async (ev: MouseEvent<HTMLButtonElement>) => {
    const codeBlock = ev.currentTarget?.closest(".block-code");
    const preElemnt = codeBlock?.querySelector('pre');
    const content = preElemnt?.textContent || "";
    await navigator.clipboard.writeText(content);
  }

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code: ({className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || "");
            const language = match ? match[1] : "";
            const isBlockCode = Boolean(language);

            if (!isBlockCode) return <code className={`inline-code ${className}`} {...props}>{children}</code>

            return <code className={`hljs ${className}`} {...props}>
              {children}
            </code>
          },
          pre: ({ children }) => {
            let language = "";
            if (React.isValidElement<{ className: string }>(children)) {
              const className = children.props.className;
              const match = /language-(\w+)/.exec(className);
              language = match ? match[1] : "";
            }

            return (
              <div className="block-code">
                <div className="title">
                  <div className="codeType">{language}</div>
                  <button type="button" className="copyButton" onClick={handleCopy}><img src={Icons.copyIcon} alt="" />复制</button>
                </div>
                <pre className="content">
                  {children}
                </pre>
              </div>
            )

          }
        }}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default MarkdownRender;
