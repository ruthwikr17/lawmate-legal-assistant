# Stage 1: Build the JAR using Maven
FROM maven:3.8.5-openjdk-17 AS build
WORKDIR /app

# Copy the pom.xml and download dependencies (caches layers)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy the source code and build it
COPY src ./src
RUN mvn clean package -DskipTests

# Stage 2: Create the minimal runtime container
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app

# Protect against potential security vulnerabilities in default alpine
RUN apk upgrade --no-cache

# Copy the built jar from Stage 1
COPY --from=build /app/target/*.jar app.jar

# Explicitly expose port 8080 for Render routing
EXPOSE 8080

# Run the jar file
ENTRYPOINT ["java", "-jar", "app.jar"]
