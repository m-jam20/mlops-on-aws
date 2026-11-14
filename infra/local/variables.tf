variable "endpoint" {
  description = "URL of the local AWS emulator"
  type        = string
  default     = "http://localhost:4566"
}

variable "project" {
  description = "Prefix for resource names"
  type        = string
  default     = "churn"
}
