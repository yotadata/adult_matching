import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { ModelLoader } from "../_shared/model_loader.ts"
import { FeaturePreprocessor } from "../_shared/feature_preprocessor.ts"

interface InferenceRequest {
  type: "user" | "item" | "both"
  user_id?: number
  video_data?: any
  user_likes?: any[]
  model_version?: string
}

interface EmbeddingResult {
  embedding: number[]
  confidence?: number
  processing_time_ms: number
}

interface InferenceResponse {
  user_embedding?: EmbeddingResult
  item_embedding?: EmbeddingResult
  similarity_score?: number
  status: "success" | "error"
  error?: string
  processing_info: {
    total_time_ms: number
    model_version: string
    timestamp: string
  }
}

serve(async (req) => {
  const startTime = performance.now()
  
  try {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    }

    if (req.method === 'OPTIONS') {
      return new Response('ok', { headers: corsHeaders })
    }

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const { type, user_id, video_data, user_likes, model_version }: InferenceRequest = await req.json()
    
    console.log(`=== Two-Tower Inference Request ===`)
    console.log(`Type: ${type}, User ID: ${user_id}, Model Version: ${model_version || 'latest'}`)

    // Initialize modules
    const modelLoader = new ModelLoader()
    const preprocessor = new FeaturePreprocessor()
    
    const response: InferenceResponse = {
      status: "success",
      processing_info: {
        total_time_ms: 0,
        model_version: model_version || 'latest',
        timestamp: new Date().toISOString()
      }
    }

    // User embedding generation
    if (type === "user" || type === "both") {
      console.log("=== Generating User Embedding ===")
      const userStartTime = performance.now()
      
      try {
        // Load user tower model
        const userTowerResult = await modelLoader.loadUserTower(model_version)
        if (!userTowerResult.success) {
          throw new Error(`Failed to load user tower: ${userTowerResult.error}`)
        }
        
        // Get user data if not provided
        let userLikesData = user_likes
        if (!userLikesData && user_id) {
          console.log(`Fetching user likes for user_id: ${user_id}`)
          const { data: likes, error } = await supabase
            .from('likes')
            .select(`
              videos (
                title,
                genre,
                maker,
                duration_minutes,
                price,
                release_date,
                tags
              )
            `)
            .eq('user_id', user_id)
            .limit(100)
          
          if (error) {
            console.error("Error fetching user likes:", error)
            throw new Error(`Failed to fetch user data: ${error.message}`)
          }
          
          userLikesData = likes || []
          console.log(`Fetched ${userLikesData.length} user likes`)
        }

        // Preprocess user features
        const userFeatures = preprocessor.extractUserFeatures(userLikesData || [])
        console.log("User features extracted:", userFeatures)
        
        // Prepare model input
        const userTensorInput = preprocessor.prepareUserTensorInput(user_id || 0, userFeatures)
        console.log("User model input prepared:", `ID: ${userTensorInput.user_id.length}D, Features: ${userTensorInput.user_features.length}D`)
        
        // Generate user embedding
        const userInput = [userTensorInput.user_id, userTensorInput.user_features]
        const userEmbedding = await userTowerResult.model.predict(userInput).data()
        console.log(`User embedding generated: ${userEmbedding.length}D vector`)
        
        const userEndTime = performance.now()
        response.user_embedding = {
          embedding: Array.from(userEmbedding),
          processing_time_ms: userEndTime - userStartTime
        }
        
      } catch (error) {
        console.error("User embedding error:", error)
        response.status = "error"
        response.error = `User embedding failed: ${error.message}`
      }
    }

    // Item embedding generation
    if (type === "item" || type === "both") {
      console.log("=== Generating Item Embedding ===")
      const itemStartTime = performance.now()
      
      try {
        // Load item tower model
        const itemTowerResult = await modelLoader.loadItemTower(model_version)
        if (!itemTowerResult.success) {
          throw new Error(`Failed to load item tower: ${itemTowerResult.error}`)
        }
        
        if (!video_data) {
          throw new Error("video_data is required for item embedding generation")
        }
        
        // Preprocess item features
        const itemFeatures = preprocessor.extractItemFeatures(video_data)
        console.log("Item features extracted:", itemFeatures ? itemFeatures.length : 0, "features")
        
        // Prepare model input
        const itemTensorInput = preprocessor.prepareItemTensorInput(video_data.id || 0, itemFeatures)
        console.log("Item model input prepared:", `ID: ${itemTensorInput.item_id.length}D, Features: ${itemTensorInput.item_features.length}D`)
        
        // Generate item embedding
        const itemInput = [itemTensorInput.item_id, itemTensorInput.item_features]
        const itemEmbedding = await itemTowerResult.model.predict(itemInput).data()
        console.log(`Item embedding generated: ${itemEmbedding.length}D vector`)
        
        const itemEndTime = performance.now()
        response.item_embedding = {
          embedding: Array.from(itemEmbedding),
          processing_time_ms: itemEndTime - itemStartTime
        }
        
      } catch (error) {
        console.error("Item embedding error:", error)
        response.status = "error"
        response.error = `Item embedding failed: ${error.message}`
      }
    }

    // Calculate similarity if both embeddings available
    if (response.user_embedding && response.item_embedding && response.status === "success") {
      console.log("=== Calculating Similarity Score ===")
      const similarity = calculateCosineSimilarity(
        response.user_embedding.embedding,
        response.item_embedding.embedding
      )
      response.similarity_score = similarity
      console.log(`Similarity score: ${similarity.toFixed(4)}`)
    }

    const endTime = performance.now()
    response.processing_info.total_time_ms = endTime - startTime
    
    console.log(`=== Inference Complete ===`)
    console.log(`Total processing time: ${response.processing_info.total_time_ms.toFixed(2)}ms`)
    console.log(`Status: ${response.status}`)

    return new Response(
      JSON.stringify(response),
      { 
        headers: { 
          ...corsHeaders, 
          "Content-Type": "application/json" 
        } 
      },
    )

  } catch (error) {
    console.error("Two-Tower inference error:", error)
    
    const errorResponse: InferenceResponse = {
      status: "error",
      error: error.message,
      processing_info: {
        total_time_ms: performance.now() - startTime,
        model_version: 'unknown',
        timestamp: new Date().toISOString()
      }
    }

    return new Response(
      JSON.stringify(errorResponse),
      { 
        status: 500,
        headers: { 
          "Content-Type": "application/json",
          'Access-Control-Allow-Origin': '*',
        } 
      },
    )
  }
})

// Helper function for cosine similarity calculation
function calculateCosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have the same length for similarity calculation")
  }
  
  let dotProduct = 0
  let normA = 0
  let normB = 0
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i]
    normA += vecA[i] * vecA[i]
    normB += vecB[i] * vecB[i]
  }
  
  const denominator = Math.sqrt(normA) * Math.sqrt(normB)
  if (denominator === 0) {
    return 0
  }
  
  return dotProduct / denominator
}