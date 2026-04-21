package com.ALRS.backend.repository;

import com.ALRS.backend.model.LawGist;
import com.ALRS.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface LawGistRepository extends JpaRepository<LawGist, Long> {
    List<LawGist> findByTargetGroup(User.UserType targetGroup);
}
