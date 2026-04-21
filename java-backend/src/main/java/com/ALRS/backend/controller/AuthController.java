package com.ALRS.backend.controller;

import com.ALRS.backend.model.User;
import com.ALRS.backend.service.AuthService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "*")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
        try {
            User user = authService.register(
                    request.getEmail(), 
                    request.getPassword(), 
                    request.getFullName(), 
                    request.getPrimaryProfile() != null ? User.UserType.valueOf(request.getPrimaryProfile().toUpperCase()) : User.UserType.GENERAL
            );
            return ResponseEntity.ok(sanitize(user));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        try {
            User user = authService.login(request.getEmail(), request.getPassword());
            return ResponseEntity.ok(sanitize(user));
        } catch (Exception e) {
            return ResponseEntity.status(401).body(Map.of("error", e.getMessage()));
        }
    }
    
    private Map<String, Object> sanitize(User user) {
        Map<String, Object> safeUser = new HashMap<>();
        safeUser.put("id", user.getId());
        safeUser.put("email", user.getEmail());
        safeUser.put("fullName", user.getFullName());
        safeUser.put("primaryProfile", user.getPrimaryProfile());
        safeUser.put("createdAt", user.getCreatedAt());
        return safeUser;
    }

    public static class RegisterRequest {
        private String email;
        private String password;
        private String fullName;
        private String primaryProfile;

        // getters/setters
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getFullName() { return fullName; }
        public void setFullName(String fullName) { this.fullName = fullName; }
        public String getPrimaryProfile() { return primaryProfile; }
        public void setPrimaryProfile(String primaryProfile) { this.primaryProfile = primaryProfile; }
    }

    public static class LoginRequest {
        private String email;
        private String password;

        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }
}
