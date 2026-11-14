# Buckets the real project will use later: raw data and trained models.
resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-data"
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.project}-models"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled" # keep every version of a model file
  }
}
