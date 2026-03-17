import * as dotenv from 'dotenv';
dotenv.config();//To access the environment variables.
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';//Utility function for importing pdf files.
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
//utility function for creating chunking of documents. (means to break the document into small pieces.)
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';//Utility function for embeding model.
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';//This utility Function does both Vector embedding and store in database.



async function indexing() {
    
    // pdf file ko load kariye
    const PDF_PATH = './NODE.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();

     
    //  chunking create karna
    

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,//ager ek context ka kuch information pichle wale chunk mai chala jata hai to uske liye overlap use karte hai. (means 200 characters ka overlap hoga.)   
    });//this object is for creating chunking.

    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

    // console.log(chunkedDocs.length); 266 chunk --> vector

    //chunks ke corresponding vector chahiye.


    // embeding create karni hai
    //Har lLM model ke khudke Embeding model hote hai.
    //We will use Embedding model of Gemini.

    // configure(Declaration) kar diya hai(MEANS SET UP KIYA HAI FOR EMBEDDING, embedding not done yet.)
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model:  'models/gemini-embedding-001',
    });

   

    // configure(declaration) pinecone( means ye mai use karunga and all.)

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);//iss .env ye index uthke aa jayenge .


    // single step--> ChunkedDocs-->Embedding-->Vector DB.
    // ( Embedding create hoga + wo Vector DB mai bhi store ho jayega. (means dono kaam ek sath ho jayega.))

    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });
}
//maxConcurrency ka matlab hai ki ek sath kitne parallel mai chunks ko process kar sakhte hai. (means 5 chunks ek sath process honge.)

//like for 500 chunks divide it by 5 , so 100 chunks in one batch and total 5 batches. (means 5 parallel process honge.)

indexing();
//Ab hoga Query Phase