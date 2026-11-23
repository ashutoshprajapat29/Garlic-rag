require("dotenv").config();

const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
});

async function main() {

    const response = await ai.models.generateContent({
        model: "gemini-3.8-flash",
        contents: "Explain in one sentence what garlic needs for good growth."
    });

    console.log(response.text);
}

main().catch(error => {
    console.error(error);
});