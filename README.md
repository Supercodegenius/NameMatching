# ReMatch | AI-Powered Insured Name Matching for Reinsurance
This repository contains ReMatch, a BrainCal application for cleaning, normalizing, and matching insured names in reinsurance bordereaux using AI-assisted and rules-based matching methods.

## Product Description

ReMatch is designed for reinsurance teams that need accurate insured-name matching at scale. The app combines multiple matching methods, AI-assisted ranking, and a workflow-focused UI to reduce manual cleanup and improve confidence in downstream exposure and reporting outputs.

## SLM Support

The repository includes a local Small Language Model (SLM) matcher powered by a bi-encoder model in `outputs/biencoder`.

- Supported SLM methods in the matcher: `SLM Match`, `Vector Similarity (SLM)`, and `SLM Adaptive Match`.
- Standalone SLM tester page is available at query route `?page=slm`.
- Model artifacts are expected under `outputs/biencoder/`.

### Optional ANN Acceleration

For faster large-scale semantic retrieval, install FAISS:

```bash
pip install faiss-cpu
```

If FAISS is not installed, SLM matching still works and falls back to non-ANN execution.

# File Structure
There is one branch for this course containing four chapter folders corresponding to the four chapters in the course. Inside these chapter folders are the coding exercise files. The naming convention for the exercise files is CHAPTER#_MOVIE#. As an example, the branch named 02_03 corresponds to the second chapter and the third video in that chapter.

All coding exercise files will have a beginning and an end state, for example '02_03e' would be the ending file for the third movie in the second chapter. These are marked with the letters b for "beginning" and e for "end". The b file contains the code as it is at the beginning of the movie. The e file contains the code as it is at the end of the movie.

## Maintainer

BrainCal Tech



