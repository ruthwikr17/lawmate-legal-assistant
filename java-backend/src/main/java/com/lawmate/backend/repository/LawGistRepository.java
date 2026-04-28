package com.lawmate.backend.repository;

import com.lawmate.backend.model.LawGist;
import com.lawmate.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface LawGistRepository extends JpaRepository<LawGist, Long> {
    List<LawGist> findByTargetGroup(User.UserType targetGroup);
}
