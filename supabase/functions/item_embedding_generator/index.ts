import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { ModelLoader } from "../_shared/model_loader.ts"
import { FeaturePreprocessor } from "../_shared/feature_preprocessor.ts"

interface BatchEmbeddingRequest {
  video_ids?: number[]
  batch_size?: number
  model_version?: string
  sync_with_database?: boolean
  force_regenerate?: boolean
}

interface SingleEmbeddingRequest {
  video_id: number
  video_data?: any
  model_version?: string
  save_to_database?: boolean
}

interface EmbeddingResponse {
  embeddings: {
    video_id: number
    embedding: number[]
    processing_time_ms: number
    confidence_score: number
  }[]
  batch_stats: {
    total_processed: number
    successful: number
    failed: number
    average_processing_time_ms: number
    total_time_ms: number
  }
  model_info: {
    version: string
    timestamp: string
    feature_dimension: number
  }
  status: "success" | "partial" | "error"
  error_details?: string[]
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

    const requestData = await req.json()
    console.log(`=== Item Embedding Generation Request ===`)
    console.log(`Request type: ${requestData.video_ids ? 'batch' : 'single'}`)
    
    // Initialize modules
    const modelLoader = new ModelLoader()
    const preprocessor = new FeaturePreprocessor()
    
    // Load item tower model
    console.log("Loading Item Tower model...")
    const itemTowerResult = await modelLoader.loadItemTower(requestData.model_version)
    if (!itemTowerResult.success) {
      throw new Error(`Failed to load item tower: ${itemTowerResult.error}`)
    }
    console.log("✅ Item Tower model loaded successfully")

    let embeddings: any[] = []
    let errorDetails: string[] = []
    let totalProcessed = 0
    
    // Batch processing
    if (requestData.video_ids) {
      console.log(`=== Batch Processing: ${requestData.video_ids.length} videos ===`)
      const batchRequest: BatchEmbeddingRequest = requestData
      const batchSize = batchRequest.batch_size || 10
      
      // Fetch video data
      console.log("Fetching video data from database...")
      const { data: videos, error } = await supabase
        .from('videos')
        .select('*')
        .in('id', batchRequest.video_ids)
      
      if (error) {
        throw new Error(`Failed to fetch videos: ${error.message}`)
      }
      
      console.log(`📹 Fetched ${videos?.length || 0} videos`)
      
      // Process in batches
      for (let i = 0; i < (videos?.length || 0); i += batchSize) {
        const batch = videos!.slice(i, i + batchSize)
        console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil((videos?.length || 0)/batchSize)}`)
        
        for (const video of batch) {
          try {
            const embeddingStartTime = performance.now()
            
            // Extract and prepare features
            const itemFeatures = preprocessor.extractItemFeatures(video)
            const itemTensorInput = preprocessor.prepareItemTensorInput(video.id, itemFeatures)
            
            // Generate embedding
            const itemInput = [itemTensorInput.item_id, itemTensorInput.item_features]
            const embeddingTensor = await itemTowerResult.model.predict(itemInput).data()
            const embedding = Array.from(embeddingTensor)
            
            const processingTime = performance.now() - embeddingStartTime
            
            // Calculate confidence score (based on feature completeness and vector magnitude)
            const vectorMagnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0))
            const featureCompleteness = itemFeatures.quality_features.data_completeness
            const confidenceScore = Math.min(0.95, (vectorMagnitude * featureCompleteness * 0.8) + 0.1)
            
            embeddings.push({
              video_id: video.id,
              embedding,
              processing_time_ms: processingTime,
              confidence_score: confidenceScore
            })
            
            // Save to database if requested
            if (batchRequest.sync_with_database) {
              const { error: updateError } = await supabase
                .from('video_embeddings')
                .upsert({
                  video_id: video.id,
                  embedding: embedding,
                  model_version: requestData.model_version || 'latest',
                  confidence_score: confidenceScore,
                  created_at: new Date().toISOString(),
                  updated_at: new Date().toISOString()
                })
              
              if (updateError) {
                console.error(`Failed to save embedding for video ${video.id}:`, updateError)
                errorDetails.push(`Database save failed for video ${video.id}: ${updateError.message}`)
              }
            }
            
            totalProcessed++
            console.log(`✅ Video ${video.id}: ${embedding.length}D embedding (confidence: ${(confidenceScore*100).toFixed(1)}%)`)
            
          } catch (error) {
            console.error(`❌ Failed to process video ${video.id}:`, error)
            errorDetails.push(`Video ${video.id}: ${error.message}`)
          }
        }
        
        // Small delay between batches to prevent overload
        if (i + batchSize < (videos?.length || 0)) {
          await new Promise(resolve => setTimeout(resolve, 100))
        }
      }
      
    } else {
      // Single video processing
      console.log("=== Single Video Processing ===")
      const singleRequest: SingleEmbeddingRequest = requestData
      
      let videoData = singleRequest.video_data
      
      // Fetch video data if not provided
      if (!videoData) {
        console.log(`Fetching video data for ID: ${singleRequest.video_id}`)
        const { data: video, error } = await supabase
          .from('videos')
          .select('*')
          .eq('id', singleRequest.video_id)
          .single()
        
        if (error) {
          throw new Error(`Failed to fetch video: ${error.message}`)
        }
        
        videoData = video
      }
      
      const embeddingStartTime = performance.now()
      
      // Process single video
      const itemFeatures = preprocessor.extractItemFeatures(videoData)
      const itemTensorInput = preprocessor.prepareItemTensorInput(singleRequest.video_id, itemFeatures)
      
      const itemInput = [itemTensorInput.item_id, itemTensorInput.item_features]
      const embeddingTensor = await itemTowerResult.model.predict(itemInput).data()
      const embedding = Array.from(embeddingTensor)
      
      const processingTime = performance.now() - embeddingStartTime
      
      const vectorMagnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0))
      const featureCompleteness = itemFeatures.quality_features.data_completeness
      const confidenceScore = Math.min(0.95, (vectorMagnitude * featureCompleteness * 0.8) + 0.1)
      
      embeddings.push({
        video_id: singleRequest.video_id,
        embedding,
        processing_time_ms: processingTime,
        confidence_score: confidenceScore
      })
      
      // Save to database if requested
      if (singleRequest.save_to_database) {
        const { error: updateError } = await supabase
          .from('video_embeddings')
          .upsert({
            video_id: singleRequest.video_id,
            embedding: embedding,
            model_version: requestData.model_version || 'latest',
            confidence_score: confidenceScore,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          })
        
        if (updateError) {
          errorDetails.push(`Database save failed: ${updateError.message}`)
        }
      }
      
      totalProcessed = 1
      console.log(`✅ Single video ${singleRequest.video_id}: ${embedding.length}D embedding (confidence: ${(confidenceScore*100).toFixed(1)}%)`)
    }

    const endTime = performance.now()
    const totalTime = endTime - startTime
    const averageTime = embeddings.length > 0 ? embeddings.reduce((sum, e) => sum + e.processing_time_ms, 0) / embeddings.length : 0
    
    const response: EmbeddingResponse = {
      embeddings,
      batch_stats: {
        total_processed: totalProcessed,
        successful: embeddings.length,
        failed: errorDetails.length,
        average_processing_time_ms: averageTime,
        total_time_ms: totalTime
      },
      model_info: {
        version: requestData.model_version || 'latest',
        timestamp: new Date().toISOString(),
        feature_dimension: 768 // Two-Tower embedding dimension
      },
      status: errorDetails.length === 0 ? "success" : 
              embeddings.length > 0 ? "partial" : "error",
      error_details: errorDetails.length > 0 ? errorDetails : undefined
    }

    console.log(`=== Embedding Generation Complete ===`)
    console.log(`✅ Successful: ${response.batch_stats.successful}`)
    console.log(`❌ Failed: ${response.batch_stats.failed}`)
    console.log(`⏱️  Total time: ${totalTime.toFixed(2)}ms`)
    console.log(`📊 Average per item: ${averageTime.toFixed(2)}ms`)

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
    console.error("Item embedding generation error:", error)
    
    const errorResponse: EmbeddingResponse = {
      embeddings: [],
      batch_stats: {
        total_processed: 0,
        successful: 0,
        failed: 1,
        average_processing_time_ms: 0,
        total_time_ms: performance.now() - startTime
      },
      model_info: {
        version: 'unknown',
        timestamp: new Date().toISOString(),
        feature_dimension: 768
      },
      status: "error",
      error_details: [error.message]
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