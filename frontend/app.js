
const API_URL = "http://127.0.0.1:8000";

const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const conversation = document.getElementById("conversation");
const topKSelect = document.getElementById("topK");
const clearChatButton = document.getElementById("clearChat");

const sourcesPanel = document.getElementById("sourcesPanel");
const sourcesList = document.getElementById("sourcesList");
const sourceCount = document.getElementById("sourceCount");


// --------------------------------------------------
// NAVEGAÇÃO ENTRE SEÇÕES
// --------------------------------------------------

function showPage(pageName) {
    document.querySelectorAll(".page").forEach((page) => {
        const active = page.id === `page-${pageName}`;

        page.classList.toggle("active", active);
        page.hidden = !active;
    });

    document.querySelectorAll("[data-page]").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.page === pageName
        );
    });
}

document.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => {
        showPage(button.dataset.page);
    });
});


// --------------------------------------------------
// MENSAGENS DO CHAT
// --------------------------------------------------

function addMessage(role, text, meta = "") {
    const wrapper = document.createElement("div");
    wrapper.className = `message message-${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    // textContent evita interpretar a resposta como HTML.
    bubble.textContent = text;

    wrapper.appendChild(bubble);

    if (meta) {
        const metadata = document.createElement("div");
        metadata.className = "message-meta";
        metadata.textContent = meta;
        wrapper.appendChild(metadata);
    }

    conversation.appendChild(wrapper);
    wrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });

    return wrapper;
}


// --------------------------------------------------
// EXIBIR FONTES RECUPERADAS
// --------------------------------------------------

function renderSources(sources) {
    sourcesList.replaceChildren();

    sourceCount.textContent =
        `${sources.length} ${sources.length === 1 ? "trecho" : "trechos"}`;

    sources.forEach((source, index) => {
        const item = document.createElement("article");
        item.className = "source-item";

        const number = document.createElement("span");
        number.className = "source-number";
        number.textContent = String(index + 1);

        const details = document.createElement("div");

        const heading = document.createElement("strong");
        heading.textContent = `Chunk ${source.chunk}`;

        const filename = document.createElement("p");
        filename.textContent = source.source;

        const distance = document.createElement("p");
        distance.className = "distance";
        distance.textContent =
            `Distância vetorial: ${Number(source.distance).toFixed(4)}`;

        details.append(heading, filename, distance);
        item.append(number, details);
        sourcesList.appendChild(item);
    });

    sourcesPanel.hidden = sources.length === 0;
}


// --------------------------------------------------
// ENVIAR PERGUNTA PARA A API
// --------------------------------------------------

async function sendQuestion(question) {
    const normalizedQuestion = question.trim();

    if (normalizedQuestion.length < 3) {
        questionInput.focus();
        return;
    }

    showPage("chat");

    addMessage("user", normalizedQuestion);
    questionInput.value = "";
    sendButton.disabled = true;
    sendButton.textContent = "Consultando...";
    sourcesPanel.hidden = true;

    const loadingMessage = addMessage(
        "assistant",
        "Consultando os documentos e gerando a resposta..."
    );

    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: normalizedQuestion,
                top_k: Number(topKSelect.value)
            })
        });

        const result = await response.json();

        if (!response.ok) {
            const detail = typeof result.detail === "string"
                ? result.detail
                : JSON.stringify(result.detail ?? result);

            throw new Error(detail || `Erro HTTP ${response.status}`);
        }

        loadingMessage.remove();

        addMessage("assistant", result.answer, "Resposta gerada pela LLM");

        renderSources(
            Array.isArray(result.sources) ? result.sources : []
        );

    } catch (error) {
        loadingMessage.remove();

        addMessage(
            "assistant",
            `Não foi possível consultar a API.\n\n${error.message}\n\n` +
            "Confira se o Uvicorn está rodando em http://127.0.0.1:8000 " +
            "e se o Ollama está disponível."
        );

        sourcesPanel.hidden = true;
        console.error("Erro na consulta:", error);

    } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Enviar ↗";
        questionInput.focus();
    }
}


// --------------------------------------------------
// EVENTOS DO FORMULÁRIO
// --------------------------------------------------

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendQuestion(questionInput.value);
});


// --------------------------------------------------
// PERGUNTAS SUGERIDAS
// --------------------------------------------------

document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
        sendQuestion(button.dataset.question);
    });
});


// --------------------------------------------------
// LIMPAR CONVERSA
// --------------------------------------------------

clearChatButton.addEventListener("click", () => {
    conversation.replaceChildren();

    sourcesList.replaceChildren();
    sourcesPanel.hidden = true;
    sourceCount.textContent = "0 trechos";

    const welcome = document.createElement("div");
    welcome.className = "welcome-message";

    const avatar = document.createElement("span");
    avatar.className = "assistant-avatar";
    avatar.textContent = "✳";

    const content = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = "Conversa reiniciada";

    const description = document.createElement("p");
    description.textContent =
        "Faça uma nova pergunta sobre os documentos científicos.";

    content.append(title, description);
    welcome.append(avatar, content);
    conversation.appendChild(welcome);

    questionInput.focus();
});


// --------------------------------------------------
// INICIALIZAÇÃO
// --------------------------------------------------

showPage("chat");
