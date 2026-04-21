package com.ALRS.backend.service;

import com.ALRS.backend.model.LawGist;
import com.ALRS.backend.model.User;
import com.ALRS.backend.repository.LawGistRepository;
import com.ALRS.backend.repository.UserRepository;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class UserRightsService {

    private final LawGistRepository lawGistRepository;
    private final UserRepository userRepository;

    public UserRightsService(LawGistRepository lawGistRepository, UserRepository userRepository) {
        this.lawGistRepository = lawGistRepository;
        this.userRepository = userRepository;
    }

    public List<LawGist> getPersonalizedRights(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Return gists specific to user profile + general ones
        List<LawGist> profileGists = lawGistRepository.findByTargetGroup(user.getPrimaryProfile());
        List<LawGist> generalGists = lawGistRepository.findByTargetGroup(User.UserType.GENERAL);
        
        profileGists.addAll(generalGists);
        return profileGists;
    }
}
