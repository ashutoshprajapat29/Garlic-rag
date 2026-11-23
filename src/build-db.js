const fs = require("fs");
const path = require("path");
const { ChromaClient } = require("chromadb");
const { pipeline } = require("@xenova/transformers");
const pdf = require("pdf-parse");

const DATA_DIR = path.join(__dirname, "..", "data");

const COLLECTION_NAME = "garlic_knowledge";

async function extractPDF(filePath) {
    const dataBuffer = fs.readFileSync(filePath);

    const parser = new pdf.PDFParse({ data: dataBuffer });

    const result = await parser.getText();

    await parser.destroy();

    return result.text;
}

function createChunks(text, chunkSize = 1000, overlap = 200) {
    const chunks = [];

    let start = 0;

    while (start < text.length) {
        const chunk = text
            .slice(start, start + chunkSize)
            .trim();

        if (chunk.length > 0) {
            chunks.push(chunk);
        }

        start += chunkSize - overlap;
    }

    return chunks;
}

async function main() {

    console.log("Loading embedding model...");

    const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    console.log("Embedding model ready.");

    // Connect to ChromaDB
    const client = new ChromaClient({
    path: "http://localhost:8000"
});

    const collection = await client.getOrCreateCollection({
        name: COLLECTION_NAME
    });

    const files = fs
        .readdirSync(DATA_DIR)
        .filter(file => file.toLowerCase().endsWith(".pdf"));

    if (files.length === 0) {
        console.log("No PDF files found in data folder.");
        return;
    }

    let idCounter = 0;

    for (const file of files) {

        console.log(`\nProcessing: ${file}`);

        const filePath = path.join(DATA_DIR, file);

        const text = await extractPDF(filePath);

        console.log(`Extracted ${text.length} characters.`);

        const chunks = createChunks(text);

        console.log(`Created ${chunks.length} chunks.`);

        const ids = [];
        const documents = [];
        const embeddings = [];
        const metadatas = [];

        for (let i = 0; i < chunks.length; i++) {

            const chunk = chunks[i];

            const output = await extractor(chunk, {
                pooling: "mean",
                normalize: true
            });

            const embedding = Array.from(output.data);

            ids.push(`chunk_${idCounter}`);

            documents.push(chunk);

            embeddings.push(embedding);

            metadatas.push({
                source: file,
                chunk: i
            });

            idCounter++;
        }

        await collection.add({
            ids,
            documents,
            embeddings,
            metadatas
        });

        console.log(`Stored ${chunks.length} chunks in ChromaDB.`);
    }

    console.log("\n================================");
    console.log("RAG DATABASE CREATED!");
    console.log("================================");

    console.log(
        "Total chunks:",
        await collection.count()
    );
}

main().catch(error => {
    console.error("\nERROR:");
    console.error(error);
});