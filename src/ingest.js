const fs = require("fs");
const path = require("path");
const { PDFParse } = require("pdf-parse");

const DATA_DIR = path.join(__dirname, "..", "data");

async function extractPDF(filePath) {
    const dataBuffer = fs.readFileSync(filePath);

    const parser = new PDFParse({ data: dataBuffer });

const result = await parser.getText();

await parser.destroy();

return result.text;
}

function createChunks(text, chunkSize = 1000, overlap = 200) {
    const chunks = [];

    let start = 0;

    while (start < text.length) {
        const end = start + chunkSize;

        const chunk = text.slice(start, end).trim();

        if (chunk.length > 0) {
            chunks.push(chunk);
        }

        start += chunkSize - overlap;
    }

    return chunks;
}

async function main() {
    const files = fs
        .readdirSync(DATA_DIR)
        .filter(file => file.toLowerCase().endsWith(".pdf"));

    if (files.length === 0) {
        console.log("No PDF files found in data folder.");
        return;
    }

    for (const file of files) {
        const filePath = path.join(DATA_DIR, file);

        console.log(`\nReading: ${file}`);

        const text = await extractPDF(filePath);

        console.log(`Extracted characters: ${text.length}`);

        const chunks = createChunks(text);

        console.log(`Created chunks: ${chunks.length}`);

        console.log("\nFirst chunk:\n");
        console.log(chunks[0]);
    }
}

main();