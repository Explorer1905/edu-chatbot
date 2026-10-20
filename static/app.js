const askBtn = document.getElementById("askBtn");
const questionEl = document.getElementById("question");
const gradeEl = document.getElementById("grade");
const subjectEl = document.getElementById("subject");
const statusMsg = document.getElementById("statusMsg");

const welcomeBlock = document.getElementById("welcomeBlock");
const conversation = document.getElementById("conversation");
const userQuestionEl = document.getElementById("userQuestion");

const resultPanel = document.getElementById("resultPanel");
const answerText = document.getElementById("answerText");
const metaGrade = document.getElementById("metaGrade");
const metaSubject = document.getElementById("metaSubject");

const gapsCard = document.getElementById("gapsCard");
const gapsChips = document.getElementById("gapsChips");

const sourcesCard = document.getElementById("sourcesCard");
const sourcesList = document.getElementById("sourcesList");

const graphCard = document.getElementById("graphCard");
const graphSvgWrap = document.getElementById("graphSvgWrap");

const focusQuestionBtn = document.getElementById("focusQuestion");
const newChatBtn = document.getElementById("newChatBtn");

const NCERT_BROWSE_URL = "https://ncert.nic.in/textbook.php";
let currentSimulation = null;

// ---------- Suggestion pills ----------
document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    const q = btn.getAttribute("data-question");
    if (q) {
      questionEl.value = q;
      questionEl.focus();
      askQuestion();
    }
  });
});

// ---------- Follow-up / new question shortcuts ----------
if (focusQuestionBtn) {
  focusQuestionBtn.addEventListener("click", () => questionEl.focus());
}
if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    resultPanel.hidden = true;
    welcomeBlock.hidden = false;
    questionEl.value = "";
    statusMsg.textContent = "";
    questionEl.focus();
  });
}

// ---------- Auto-grow textarea ----------
questionEl.addEventListener("input", () => {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 160) + "px";
});

// ---------- Enter to send (Shift+Enter for newline) ----------
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

askBtn.addEventListener("click", askQuestion);

async function askQuestion() {
  const question = questionEl.value.trim();
  if (!question) {
    statusMsg.textContent = "Please type a question first.";
    statusMsg.className = "status error";
    return;
  }

  askBtn.disabled = true;
  statusMsg.textContent = "Thinking... this can take 10-20 seconds.";
  statusMsg.className = "status";
  resultPanel.hidden = true;
  if (welcomeBlock) welcomeBlock.hidden = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        grade: gradeEl.value || null,
        subject: subjectEl.value || null,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      statusMsg.textContent = data.error || "Something went wrong.";
      statusMsg.className = "status error";
      if (welcomeBlock) welcomeBlock.hidden = false;
      return;
    }

    statusMsg.textContent = "";
    if (userQuestionEl) userQuestionEl.textContent = question;
    renderResult(data);
  } catch (err) {
    statusMsg.textContent = "Could not reach the server. Is app.py running?";
    statusMsg.className = "status error";
    if (welcomeBlock) welcomeBlock.hidden = false;
  } finally {
    askBtn.disabled = false;
  }
}

function renderResult(data) {
  answerText.innerHTML = marked.parse(cleanMathNotation(data.answer));

  metaGrade.textContent = data.detected_grade ? `Class ${data.detected_grade}` : "Class: n/a";
  metaSubject.textContent = data.detected_subject || "Subject: n/a";

  // Gaps
  if (data.gaps && data.gaps.length) {
    gapsChips.innerHTML = "";
    data.gaps.forEach((g) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = g;
      gapsChips.appendChild(chip);
    });
    gapsCard.hidden = false;
  } else {
    gapsCard.hidden = true;
  }

  // Sources, with a link to NCERT's browse page
  if (data.sources && data.sources.length) {
    sourcesList.innerHTML = "";
    data.sources.forEach((s) => {
      const li = document.createElement("li");
      const left = document.createElement("div");
      left.innerHTML = `<span class="source-topic">${escapeHtml(s.topic)}</span>` +
        `<span class="source-meta">${escapeHtml(s.subject)}, Class ${s.grade}</span>`;

      const link = document.createElement("a");
      link.href = NCERT_BROWSE_URL;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "source-link";
      link.textContent = "Find in NCERT textbooks";

      li.appendChild(left);
      li.appendChild(link);
      sourcesList.appendChild(li);
    });
    sourcesCard.hidden = false;
  } else {
    sourcesCard.hidden = true;
  }

  // Graph
  if (data.graph && data.graph.nodes && data.graph.nodes.length > 1) {
    drawGraph(data.graph);
    graphCard.hidden = false;
  } else {
    graphCard.hidden = true;
  }

  resultPanel.hidden = false;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawGraph(graph) {
  graphSvgWrap.innerHTML = "";

  const width = graphSvgWrap.clientWidth || 640;
  const height = 420;
  const nodes = graph.nodes.map((node) => ({ ...node }));
  const links = graph.edges.map((edge) => ({ source: edge.source, target: edge.target }));

  const svg = d3.select(graphSvgWrap)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "img")
    .attr("aria-label", "Interactive curriculum graph");

  if (currentSimulation) currentSimulation.stop();
  redrawWithData(nodes, links, width, height, svg);
}

function dragBehavior(simulation) {
  return d3.drag()
    .on("start", (event, node) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      node.fx = node.x;
      node.fy = node.y;
    })
    .on("drag", (event, node) => {
      node.fx = event.x;
      node.fy = event.y;
    })
    .on("end", (event, node) => {
      if (!event.active) simulation.alphaTarget(0);
      node.fx = null;
      node.fy = null;
    });
}

async function handleNodeClick(node, nodes, links, width, height) {
  if (node.type === "source") {
    window.open(NCERT_BROWSE_URL, "_blank", "noopener,noreferrer");
    return;
  }
  if (node.type === "question" || node.expanded) return;

  node.expanded = true;
  try {
    const response = await fetch("/expand_concept", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ concept: node.label }),
    });
    if (!response.ok) throw new Error("Could not expand concept");
    const data = await response.json();
    const existingIds = new Set(nodes.map((item) => item.id));
    const newNodes = data.nodes.filter((item) => item.id !== node.id && !existingIds.has(item.id));
    const newLinks = data.edges.filter((edge) => !links.some((link) => link.source.id === edge.source && link.target.id === edge.target));

    if (!newNodes.length) {
      node.expanded = false;
      return;
    }
    newNodes.forEach((item) => {
      item.x = node.x + (Math.random() - 0.5) * 40;
      item.y = node.y + (Math.random() - 0.5) * 40;
    });
    nodes.push(...newNodes);
    links.push(...newLinks.map((edge) => ({ source: edge.source, target: edge.target })));
    redrawWithData(nodes, links, width, height);
  } catch (error) {
    node.expanded = false;
  }
}

function redrawWithData(nodes, links, width, height, existingSvg = null) {
  const svg = existingSvg || d3.select(graphSvgWrap).select("svg");
  svg.selectAll("*").remove();
  if (currentSimulation) currentSimulation.stop();

  const linkGroup = svg.append("g").attr("class", "links");
  const nodeGroup = svg.append("g").attr("class", "nodes");
  currentSimulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((node) => node.id).distance(90).strength(0.6))
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(42));

  const link = linkGroup.selectAll("line").data(links).join("line").attr("class", "graph-edge");
  const node = nodeGroup.selectAll("g")
    .data(nodes)
    .join("g")
    .attr("class", (item) => `graph-node node-${item.type === "question" ? "q" : item.type === "gap" ? "gap" : "src"}`)
    .style("cursor", "pointer")
    .call(dragBehavior(currentSimulation));

  node.append("circle").attr("r", (item) => item.type === "question" ? 11 : 8);
  node.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", (item) => item.type === "question" ? 26 : 22)
    .selectAll("tspan")
    .data((item) => wrapLabel(item.label).map((line) => ({ line, node: item })))
    .join("tspan")
    .attr("x", 0)
    .attr("dy", (item, index) => index === 0 ? 0 : 12)
    .text((item) => item.line);
  node.on("click", (event, item) => handleNodeClick(item, nodes, links, width, height));

  currentSimulation.on("tick", () => {
    link.attr("x1", (edge) => edge.source.x).attr("y1", (edge) => edge.source.y).attr("x2", (edge) => edge.target.x).attr("y2", (edge) => edge.target.y);
    node.attr("transform", (item) => `translate(${item.x},${item.y})`);
  });
}

function wrapLabel(label) {
  const maxCharsPerLine = 16;
  const lines = [];
  let current = "";
  label.split(" ").forEach((word) => {
    if ((current + " " + word).trim().length > maxCharsPerLine) {
      if (current) lines.push(current.trim());
      current = word;
    } else {
      current = (current + " " + word).trim();
    }
  });
  if (current) lines.push(current);
  return lines.slice(0, 2);
}

function cleanMathNotation(text) {
  return text
    .replace(/\\\[(.+?)\\\]/gs, (_, expr) => `\n\n${formatMathExpr(expr)}\n\n`)
    .replace(/\\\((.+?)\\\)/gs, (_, expr) => formatMathExpr(expr));
}

function formatMathExpr(expr) {
  return expr
    .replace(/\\alpha/g, "α")
    .replace(/\\beta/g, "β")
    .replace(/\\Delta/g, "Δ")
    .replace(/\\times/g, "×")
    .replace(/\\cdot/g, "·")
    .replace(/\^\{?(-?\d+)\}?/g, (_, value) => toSuperscript(value))
    .replace(/_\{?(\w+)\}?/g, (_, value) => toSubscript(value))
    .replace(/\\,/g, " ")
    .replace(/\\;/g, " ")
    .trim();
}

const SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const SUB = { "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉" };

function toSuperscript(value) { return value.split("").map((char) => SUP[char] || char).join(""); }
function toSubscript(value) { return value.split("").map((char) => SUB[char] || char).join(""); }

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}