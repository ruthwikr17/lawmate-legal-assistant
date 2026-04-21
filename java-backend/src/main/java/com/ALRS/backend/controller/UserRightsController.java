package com.ALRS.backend.controller;

import com.ALRS.backend.model.LawGist;
import com.ALRS.backend.service.UserRightsService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/rights")
@CrossOrigin(origins = "*")
public class UserRightsController {

    private final UserRightsService userRightsService;

    public UserRightsController(UserRightsService userRightsService) {
        this.userRightsService = userRightsService;
    }

    @GetMapping("/{userId}")
    public ResponseEntity<List<LawGist>> getPersonalizedRights(@PathVariable Long userId) {
        return ResponseEntity.ok(userRightsService.getPersonalizedRights(userId));
    }
}
