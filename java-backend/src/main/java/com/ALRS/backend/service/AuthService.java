package com.ALRS.backend.service;

import com.ALRS.backend.model.User;
import com.ALRS.backend.repository.UserRepository;
import org.springframework.stereotype.Service;
import java.security.MessageDigest;
import java.util.Base64;
import java.util.Optional;

@Service
public class AuthService {
    
    private final UserRepository userRepository;

    public AuthService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User register(String email, String rawPassword, String fullName, User.UserType primaryProfile) {
        if (userRepository.findByEmail(email).isPresent()) {
            throw new RuntimeException("Email already exists");
        }
        
        String hashedPassword = hashPassword(rawPassword);
        User user = User.builder()
                .email(email)
                .password(hashedPassword)
                .fullName(fullName)
                .primaryProfile(primaryProfile)
                .build();
                
        return userRepository.save(user);
    }

    public User login(String email, String rawPassword) {
        Optional<User> optionalUser = userRepository.findByEmail(email);
        if (optionalUser.isEmpty()) {
            throw new RuntimeException("Invalid email or password");
        }
        
        User user = optionalUser.get();
        if (!user.getPassword().equals(hashPassword(rawPassword))) {
            throw new RuntimeException("Invalid email or password");
        }
        
        return user;
    }

    private String hashPassword(String password) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(password.getBytes("UTF-8"));
            return Base64.getEncoder().encodeToString(hash);
        } catch (Exception e) {
            throw new RuntimeException("Failed to hash password", e);
        }
    }
}
