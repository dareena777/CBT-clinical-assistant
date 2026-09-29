document.addEventListener("DOMContentLoaded", () => {
    // Matched exact IDs from index.html
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    if (!chatForm || !userInput || !chatBox) {
        console.error("One or more chat UI elements were not found in the DOM.");
        return;
    }

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const messageText = userInput.value.trim();
        if (!messageText) return;

        // 1. Render User Message
        appendMessage("user-message", messageText);
        userInput.value = "";

        // 2. Render Assistant Placeholder
        const assistantMessageElem = appendMessage("assistant-message", "");

        try {
            // 3. POST to /api/chat endpoint
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: messageText,
                    session_id: "default_user"
                }),
            });

            if (!response.ok) {
                assistantMessageElem.innerText = "Error: Unable to process request.";
                return;
            }

            // 4. Stream response chunk-by-chunk
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                assistantMessageElem.innerText += chunk;
                chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll
            }
        } catch (error) {
            console.error("Streaming error:", error);
            assistantMessageElem.innerText = "Connection error. Please try again.";
        }
    });

    function appendMessage(senderClass, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${senderClass}`;
        msgDiv.innerText = text;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    }
});