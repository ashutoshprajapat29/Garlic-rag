const { pipeline } = require("@xenova/transformers");

function cosineSimilarity(a, b) {
    let dot = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magnitudeA += a[i] * a[i];
        magnitudeB += b[i] * b[i];
    }

    return dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
}

async function main() {
    const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    const texts = [
        "Garlic needs fertile soil for good growth.",
        "Which soil is suitable for growing garlic?",
        "The weather today is rainy."
    ];

    const embeddings = [];

    for (const text of texts) {
        const output = await extractor(text, {
            pooling: "mean",
            normalize: true
        });

        embeddings.push(Array.from(output.data));
    }

    console.log("\nSimilarity between sentence 1 and 2:");

    console.log(
        cosineSimilarity(embeddings[0], embeddings[1])
    );

    console.log("\nSimilarity between sentence 1 and 3:");

    console.log(
        cosineSimilarity(embeddings[0], embeddings[2])
    );
}

main();