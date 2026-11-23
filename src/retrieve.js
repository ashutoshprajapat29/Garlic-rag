const readline = require("readline");
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});
require("dotenv").config();

const { GoogleGenAI } = require("@google/genai");
const { pipeline } = require("@xenova/transformers");
const { ChromaClient } = require("chromadb");

const client = new ChromaClient({
    host: "localhost",
    port: 8000
});

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
});

async function main() {
    const collection = await client.getCollection({
    name: "garlic_knowledge",
    embeddingFunction: null
});

    console.log("Connected to garlic knowledge base!");

    const count = await collection.count();

    console.log("Number of stored chunks:", count);
    const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
);

const question = await new Promise((resolve) => {
    rl.question("Ask a question: ", resolve);
});

const output = await extractor(question, {
    pooling: "mean",
    normalize: true
});

const questionEmbedding = Array.from(output.data);
const results = await collection.query({
    queryEmbeddings: [questionEmbedding],
    nResults: 3
});

console.log("\nTop 3 relevant chunks:\n");


const context = results.documents[0].join("\n\n");
console.log("\nContext sent to Gemini:\n");

console.log(context);

const prompt = `
You are a helpful garlic farming assistant.

Answer the user's question using the provided context.

If the answer is not present in the context, say:
"I could not find this information in the provided documents."

Context:
${context}

Question:
${question}
`;

const response = await ai.models.generateContent({
    model: "gemini-3.8-flash",
    contents: prompt
});

console.log("\nGemini Answer:\n");
console.log(response.text);

console.log("Question embedding dimensions:", questionEmbedding.length);
}


main().catch(error => {
    console.error(error);
});