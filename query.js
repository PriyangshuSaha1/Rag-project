import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI } from '@langchain/google-genai';//Utility function for embeding model and LLM model.
import { Pinecone } from '@pinecone-database/pinecone';//Utility FUnction for Pinecone database.
import * as dotenv from 'dotenv';
dotenv.config();//To access the environment variables.
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

//vector helps in comparing the semantic meaning of the data.

// configuration for embedding and LLM model
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-embedding-001',
});

//LLM model ki configuration karni hai.Sab kuch langchain ke andar hi milega, mujhe bas usko import karna hai aur use karna hai.
const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-2.5-flash',  
    temperature: 0.3, 
});

// configure Pinecone
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);



async function chatting(question) {
    

    // intent model ko introduce: Homework


    // question ki embedding create karna hai
    const queryVector = await embeddings.embedQuery(question);  

    // embeddig aagyi, uske baad usko vectorDB ke andar search karna, top10 results chahiye humein, jo question ke sabse similar ho.
    const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
    });

    // search results ke andar matches aayenge, unke andar metadata hoga, usme text hoga, us text ko extract karna hai, aur usko context ke form mai LLM ko dena hai.
    const context = searchResults.matches
                   .map(match => match.metadata.text)
                   .join("\n\n---\n\n");


    // console.log(searchResults);


    // top10+question isko mein llm ko de dunga

    const promptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on the provided documentation.

Context from the documentation:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- If the answer is not in the context, say "I don't have enough information to answer that question."
- Be concise and clear
- Use code examples from the context if relevant

Answer:
        `);

        //baat chit karne ke liye chain create karna padega, jisme prompt template, model aur output parser hoga.

        const chain = RunnableSequence.from([
            promptTemplate,
            model,
            new StringOutputParser(),
        ]);

        //await means woh wale task karne mai thora time lagega.


        //message bhej diya LLM ko, aur answer leke aaya LLM se.
        const answer = await chain.invoke({
            context: context,
            question: question,
        }); 
       

        console.log(answer);


    // LLM jo reply diya usse Output create kar dunga
}


async function main(){
   const userProblem = readlineSync.question("Ask me anything--> ");
   await chatting(userProblem);
   main();
}

//main function ko call karna hai taki user se question leke uska answer de saku.




main();