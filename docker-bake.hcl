variable "COMMIT_SHA" {
  default = "latest"
}

group "default" {
  targets = ["pocket-tts-server", "pocket-tts-app"]
}

target "pocket-tts-server" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["pocket-tts-server:${COMMIT_SHA}"]
}

target "pocket-tts-app" {
  context = "."
  dockerfile = "Dockerfile.app"
  platforms = ["linux/amd64"]
  tags = ["pocket-tts-app:${COMMIT_SHA}"]
}
