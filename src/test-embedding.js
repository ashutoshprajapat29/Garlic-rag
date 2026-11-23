const { pipeline } = require("@xenova/transformers");

async function main() {
    console.log("Loading embedding model...");

    const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    console.log("Model loaded!");

    const text = "Garlic grows well in fertile loamy soil.";

    const output = await extractor(text, {
        pooling: "mean",
        normalize: true
    });

    console.log("Embedding generated!");

    console.log("Vector dimensions:", output.data.length);

    console.log("First 10 values:");

    console.log(Array.from(output.data.slice(0, 10)));
}

main();