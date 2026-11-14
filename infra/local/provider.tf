# Points the AWS provider at a local Moto server instead of real AWS.
# Start it first with: make aws-local-up
provider "aws" {
  region     = "eu-central-1"
  access_key = "test" # fake credentials, only valid for the local emulator
  secret_key = "test"

  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true

  endpoints {
    s3  = var.endpoint
    sts = var.endpoint
  }
}
