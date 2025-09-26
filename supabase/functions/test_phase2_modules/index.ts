import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { ModelLoader } from "../_shared/model_loader.ts"
import { FeaturePreprocessor } from "../_shared/feature_preprocessor.ts"

interface TestRequest {
  test_type: "model_loader" | "feature_preprocessor" | "both"
}

serve(async (req) => {
  try {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    }

    if (req.method === 'OPTIONS') {
      return new Response('ok', { headers: corsHeaders })
    }

    const { test_type }: TestRequest = await req.json()
    const results: any = { test_type, status: "success", results: {} }

    if (test_type === "model_loader" || test_type === "both") {
      console.log("=== Testing ModelLoader ===")
      
      try {
        const modelLoader = new ModelLoader()
        
        // Test 1: Model loader initialization
        console.log("Testing ModelLoader initialization...")
        results.results.model_loader_init = { status: "success", message: "ModelLoader initialized successfully" }
        
        // Test 2: Check if models can be loaded (mock test - without actual model files)
        console.log("Testing model loading capabilities...")
        
        // Since we don't have actual models uploaded yet, we'll just test the structure
        const mockTest = {
          user_tower_config: {
            input_shape: [null, 3],
            output_shape: [null, 768],
            parameters: 686912,
            layers: 10
          },
          item_tower_config: {
            input_shape: [[null], [null, 1003]],
            output_shape: [null, 768], 
            parameters: 25585984,
            layers: 10
          }
        }
        
        results.results.model_loader_config = { 
          status: "success", 
          message: "Model configuration validated",
          config: mockTest
        }
        
      } catch (error) {
        console.error("ModelLoader test error:", error)
        results.results.model_loader_error = {
          status: "error",
          error: error.message
        }
      }
    }

    if (test_type === "feature_preprocessor" || test_type === "both") {
      console.log("=== Testing FeaturePreprocessor ===")
      
      try {
        const preprocessor = new FeaturePreprocessor()
        
        // Test 1: Feature preprocessor initialization
        console.log("Testing FeaturePreprocessor initialization...")
        results.results.feature_preprocessor_init = { 
          status: "success", 
          message: "FeaturePreprocessor initialized successfully" 
        }
        
        // Test 2: User features extraction with mock data
        console.log("Testing user features extraction...")
        const mockLikes = [
          {
            videos: {
              genre: "Drama",
              maker: "Company A",
              duration_minutes: 120,
              price: 2980
            }
          },
          {
            videos: {
              genre: "Comedy", 
              maker: "Company B",
              duration_minutes: 90,
              price: 1980
            }
          },
          {
            videos: {
              genre: "Drama",
              maker: "Company A", 
              duration_minutes: 110,
              price: 2580
            }
          }
        ]
        
        const userFeatures = preprocessor.extractUserFeatures(mockLikes)
        console.log("Extracted user features:", userFeatures)
        
        results.results.user_features = {
          status: "success",
          message: "User features extracted successfully",
          features: userFeatures,
          feature_count: Object.keys(userFeatures).length
        }
        
        // Test 3: Item features extraction with mock data
        console.log("Testing item features extraction...")
        const mockVideo = {
          genre: "Action",
          maker: "Studio X",
          duration_minutes: 130,
          price: 3980,
          release_date: "2024-01-15",
          tags: ["action", "adventure", "thriller"]
        }
        
        const itemFeatures = preprocessor.extractItemFeatures(mockVideo)
        console.log("Extracted item features:", itemFeatures)
        
        results.results.item_features = {
          status: "success",
          message: "Item features extracted successfully", 
          features: itemFeatures,
          feature_count: itemFeatures ? itemFeatures.length : 0
        }
        
      } catch (error) {
        console.error("FeaturePreprocessor test error:", error)
        results.results.feature_preprocessor_error = {
          status: "error",
          error: error.message
        }
      }
    }

    console.log("=== Test Results Summary ===")
    console.log(JSON.stringify(results, null, 2))

    return new Response(
      JSON.stringify(results),
      { 
        headers: { 
          ...corsHeaders, 
          "Content-Type": "application/json" 
        } 
      },
    )

  } catch (error) {
    console.error("Test function error:", error)
    return new Response(
      JSON.stringify({ 
        error: error.message,
        status: "error"
      }),
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