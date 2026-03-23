package com.lawmate.backend.controller;

import com.lawmate.backend.service.AIServiceClient;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.*;
import java.io.Serializable;

@RestController
@RequestMapping("/api/legal")
@CrossOrigin(origins = "*")
public class LegalChatController {

    private final AIServiceClient aiServiceClient;

    public LegalChatController(AIServiceClient aiServiceClient) {
        this.aiServiceClient = aiServiceClient;
    }

    @PostMapping("/chat")
    public ResponseEntity<AIServiceClient.AIResponse> chat(
            @RequestBody ChatRequestDTO request) {
        
        AIServiceClient.AIResponse response = aiServiceClient.chat(
                request.getQuery(), 
                request.getHistory(),
                request.getContext()
        );
        
        return ResponseEntity.ok(response);
    }

    @PostMapping("/generate")
    public ResponseEntity<AIServiceClient.AIResponse> generate(
            @RequestBody AIServiceClient.ChatRequest request) {
        
        AIServiceClient.AIResponse response = aiServiceClient.generate(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/analyze-doc")
    public ResponseEntity<Map<String, Object>> analyzeDoc(
            @RequestParam("file") org.springframework.web.multipart.MultipartFile file) {
        
        Map<String, Object> result = aiServiceClient.analyzeDoc(file);
        return ResponseEntity.ok(result);
    }

    public static class ChatRequestDTO {
        private String query;
        private List<Map<String, String>> history;
        private Map<String, Object> context;

        public String getQuery() { return query; }
        public void setQuery(String query) { this.query = query; }

        public List<Map<String, String>> getHistory() { return history; }
        public void setHistory(List<Map<String, String>> history) { this.history = history; }

        public Map<String, Object> getContext() { return context; }
        public void setContext(Map<String, Object> context) { this.context = context; }
    }
}
