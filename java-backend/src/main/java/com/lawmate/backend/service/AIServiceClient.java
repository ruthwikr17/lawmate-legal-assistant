package com.lawmate.backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.util.*;

@Service
public class AIServiceClient {

    private final RestTemplate restTemplate;

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    public AIServiceClient() {
        org.springframework.http.client.SimpleClientHttpRequestFactory factory = new org.springframework.http.client.SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(60000); // 60 seconds for OCR
        this.restTemplate = new RestTemplate(factory);
    }

    public AIResponse chat(String query, List<Map<String, String>> history, Map<String, Object> context) {
        String url = aiServiceUrl + "/chat";
        
        ChatRequest request = new ChatRequest();
        request.setQuery(query);
        request.setHistory(history);
        request.setContext(context);

        return restTemplate.postForObject(url, request, AIResponse.class);
    }

    public AIResponse generate(ChatRequest request) {
        String url = aiServiceUrl + "/generate";
        return restTemplate.postForObject(url, request, AIResponse.class);
    }

    public Map<String, Object> analyzeDoc(org.springframework.web.multipart.MultipartFile file) {
        String url = aiServiceUrl + "/analyze-doc";
        
        org.springframework.http.HttpHeaders headers = new org.springframework.http.HttpHeaders();
        // Do NOT set Content-Type manually for multipart, RestTemplate handles boundary automatically
        
        org.springframework.util.MultiValueMap<String, Object> body = new org.springframework.util.LinkedMultiValueMap<>();
        
        try {
            final String fileName = file.getOriginalFilename();
            final byte[] bytes = file.getBytes();
            
            org.springframework.core.io.ByteArrayResource resource = new org.springframework.core.io.ByteArrayResource(bytes) {
                @Override
                public String getFilename() {
                    return fileName;
                }
            };
            
            body.add("file", resource);
        } catch (java.io.IOException e) {
            throw new RuntimeException("Failed to read file for analysis", e);
        }
        
        org.springframework.http.HttpEntity<org.springframework.util.MultiValueMap<String, Object>> requestEntity = 
            new org.springframework.http.HttpEntity<>(body, headers);
            
        return restTemplate.postForObject(url, requestEntity, Map.class);
    }

    public static class ChatRequest {
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

    public static class AIResponse {
        private String answer;
        private List<SourceDTO> sources;

        public String getAnswer() { return answer; }
        public void setAnswer(String answer) { this.answer = answer; }

        public List<SourceDTO> getSources() { return sources; }
        public void setSources(List<SourceDTO> sources) { this.sources = sources; }
    }

    public static class SourceDTO {
        private String title;
        private String text;
        private Double score;
        private String act;
        private String section;

        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }

        public String getText() { return text; }
        public void setText(String text) { this.text = text; }

        public Double getScore() { return score; }
        public void setScore(Double score) { this.score = score; }

        public String getAct() { return act; }
        public void setAct(String act) { this.act = act; }

        public String getSection() { return section; }
        public void setSection(String section) { this.section = section; }
    }
}
