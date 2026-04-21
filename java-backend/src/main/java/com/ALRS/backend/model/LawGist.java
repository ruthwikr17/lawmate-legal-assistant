package com.ALRS.backend.model;

import jakarta.persistence.*;

@Entity
@Table(name = "law_gists")
public class LawGist {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String title;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String content;

    @Enumerated(EnumType.STRING)
    private User.UserType targetGroup;

    private String jurisdiction; // Central, State name, or City name

    public LawGist() {}

    public LawGist(Long id, String title, String content, User.UserType targetGroup, String jurisdiction) {
        this.id = id;
        this.title = title;
        this.content = content;
        this.targetGroup = targetGroup;
        this.jurisdiction = jurisdiction;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }

    public User.UserType getTargetGroup() { return targetGroup; }
    public void setTargetGroup(User.UserType targetGroup) { this.targetGroup = targetGroup; }

    public String getJurisdiction() { return jurisdiction; }
    public void setJurisdiction(String jurisdiction) { this.jurisdiction = jurisdiction; }

    // Simple Builder
    public static LawGistBuilder builder() {
        return new LawGistBuilder();
    }

    public static class LawGistBuilder {
        private Long id;
        private String title;
        private String content;
        private User.UserType targetGroup;
        private String jurisdiction;

        public LawGistBuilder id(Long id) { this.id = id; return this; }
        public LawGistBuilder title(String title) { this.title = title; return this; }
        public LawGistBuilder content(String content) { this.content = content; return this; }
        public LawGistBuilder targetGroup(User.UserType targetGroup) { this.targetGroup = targetGroup; return this; }
        public LawGistBuilder jurisdiction(String jurisdiction) { this.jurisdiction = jurisdiction; return this; }

        public LawGist build() {
            return new LawGist(id, title, content, targetGroup, jurisdiction);
        }
    }
}
