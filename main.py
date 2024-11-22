from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from typing import List
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from fastapi.middleware.cors import CORSMiddleware
from a2wsgi import ASGIMiddleware

# Initialize FastAPI app
app = FastAPI()
origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
wsgi_app = ASGIMiddleware(app)

# Load pre-trained model for embedding
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# CFR points and corresponding descriptions
cfr_points = [
    "Validation of systems to ensure accuracy, reliability, consistent intended performance, and the ability to discern invalid or altered records.",
    "The ability to generate accurate and complete copies of records in both human-readable and electronic form suitable for inspection, review, and copying by the agency.",
    "Protection of records to enable their accurate and ready retrieval throughout the records retention period.",
    "Limiting system access to authorized individuals.",
    "Use of secure computer-generated time-stamped audit trails to independently record the date and time of operator entries and actions that create, modify, or delete electronic records. Record changes shall not obscure previously recorded information. Such audit trail documentation shall be retained for a period at least if that required for the subject electronic records and shall be available for agency review and copying.",
    "Use of operational system checks to enforce permitted sequencing of steps and events as appropriate.",
    "Use of authority checks to ensure that only authorized individuals can use the system, electronically sign a record, access the operation or computer system input or output device, alter a record, or perform the operation at hand.",
    "Use of device (e.g., terminal) checks to determine as appropriate the validity of the source of data input or operational instruction.",
    "Determination that persons who develop, maintain, or use electronic record/electronic signature systems have the education, training, and experience to perform their assigned tasks.",
    "The establishment of and adherence to written policies that hold individuals accountable and responsible for actions initiated under their electronic signatures to deter record and signature falsification.",
    "Adequate controls over the distribution of, access to, and use of documentation for system operation and maintenance.",
    "Revision and change control procedures to maintain an audit trail that documents time-sequenced development and modification of systems documentation.",
    "Persons who use open systems to create, modify, maintain, or transmit electronic records shall employ procedures and controls designed to ensure the authenticity, integrity, and, as appropriate, confidentiality of electronic records from the point of their creation to the point of their receipt.",
    "The printed name of the signer.",
    "The date and time when the signature was executed.",
    "The meaning (such as review, approval, responsibility, or authorship) associated with the signature.",
    "Electronic signatures and handwritten signatures executed to electronic records shall be linked to their respective electronic records to ensure that the signatures cannot be excised, copied, or otherwise transferred to falsify an electronic record by ordinary means."
]

eu_annex = [
    "Does the records are in human readable form and accessible?",
    "Does the records are able to print?",
    "Does the records are able to back up?",
    "Does the retrieving process of the record retrieve the whole record information?",
    "Does the system detect invalid or altered results",
    "Does the integrity of the whole record is maintained throughout the entire retention period?",
    "Prohibit access of unauthorized individuals",
    "Access is unique to an individual",
    "Creating/modifying/deleting user access must be recorded",
    "Prevention of auto-logins",
    "Access levels are defined",
    "System administrators with the capability to delete records are independent from system users",
    "Controls for unattended systems (e.g. auto logouts)",
    "Passwords are changed",
    "Lockout or notification after X number of unsuccessful login or signature attempts",
    "Authentication of interfaces prior to transfer of data",
    "Checks for critical manually entered data (e.g. second operator or by electronic means)",
    "Does the changes are documented?",
    "Changes do not obscure previously recorded records",
    "Must be able to detect which records have changed",
    "Capture human actions that create, modify or delete records",
    "System generated",
    "Contains user ID, date/time of the action (create/modify/delete) and for changed records reason for change.",
    "Users cannot alter the date/time of the system",
    "Audit trail protected from modification and cannot be disabled",
    "System provides access to review the audit trails and when a record is critical or subject to alteration, a review is done for these records on a periodic basis to ensure data integrity",
    "Electronic signature is linked to electronic records",
    "Electronic signatures under same controls as electronic records",
    "Cannot reuse an electronic signature from another 21CFR Part 11 EU Annex 11 individual",
    "An electronic signature must have a combination of at least 2 unique components and one must only be known by the user",
    "Electronic signature shows name of the user, date/time of the signing, and meaning of signature",
    "The signature session must end after a specified period of inactivity",
    "The system reports unauthorized attempts to sign a document",
    "No automated executions of signatures",
    "Data must be encrypted",
    "Digital signatures used",
    "There must be security at both the sending and receiving systems",
    "Final revisions of paper and electronic records must have the same electric content",
    "Traceability  between electronic and paper records",
    "Electronic record must have an indication when handwritten signatures are applied to the contents of the record",
    "When paper records are produced from a database, there must be sufficient information regarding the generation of the printout so it can be reproduced",
    "Control in place to ensure all data is kept including failed result",
    "Like in paper records, those printed on thermal paper have a mechanism in place to protect them over time",
    "If successive operations, events, and/or data entry are required, the system must ensure the steps are followed in the correct sequence.",
    "For systems supporting critical processes (e.g. systems that are part of site business continuity plan), provisions must be made to ensure continuity of support for those processes in the event of a system breakdown (e.g. a manual or alternative system). The time required to bring the alternative arrangements into use must be based on risk and appropriate for a particular system and the business process it supports. These arrangements must be adequately 21CFR Part 11 EU Annex 11 documented and tested."
    
]

# Route to process the input via POST request
@app.post("/cfr")
async def process(urs_points: str = Form(...)):
    if not urs_points:
        raise HTTPException(status_code=400, detail="No URS points provided")

    # Parse the urs_input as needed
    urs_lines = [line.strip() for line in urs_points.split('\n') if line.strip()]

    # Initialize lists to store URS points and their descriptions
    urs_ids = []
    urs_descriptions = []

    # Process each line to extract URS points and descriptions
    for line in urs_lines:
        parts = line.split('-', 1)
        if len(parts) == 2:
            urs_id = parts[0].strip()
            urs_description = parts[1].strip()
            urs_ids.append(urs_id)
            urs_descriptions.append(urs_description)

    # Generate embeddings for URS and CFR points
    urs_embeddings = model.encode(urs_descriptions, convert_to_tensor=True)
    cfr_embeddings = model.encode(cfr_points, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_matrix = F.cosine_similarity(urs_embeddings.unsqueeze(1), cfr_embeddings.unsqueeze(0), dim=-1)

    # Create a mapping defaultdict
    mapping = defaultdict(list)

    # Iterate over the similarity matrix and map each URS point to the most similar CFR point
    for i, urs_id in enumerate(urs_ids):
        max_score_idx = similarity_matrix[i].argmax().item()
        mapping[cfr_points[max_score_idx]].append(urs_id)

    # Format the mapping result
    result = [{"CFR Point": cfr, "URS Reference IDs": ", ".join(urs_ids)} for cfr, urs_ids in mapping.items()]

    # Return the result as JSON
    return result

@app.post("/eu")
async def process(urs_points: str = Form(...)):
    if not urs_points:
        raise HTTPException(status_code=400, detail="No URS points provided")

    # Parse the urs_input as needed
    urs_lines = [line.strip() for line in urs_points.split('\n') if line.strip()]

    # Initialize lists to store URS points and their descriptions
    urs_ids = []
    urs_descriptions = []

    # Process each line to extract URS points and descriptions
    for line in urs_lines:
        parts = line.split('-', 1)
        if len(parts) == 2:
            urs_id = parts[0].strip()
            urs_description = parts[1].strip()
            urs_ids.append(urs_id)
            urs_descriptions.append(urs_description)

    # Generate embeddings for URS and CFR points
    urs_embeddings = model.encode(urs_descriptions, convert_to_tensor=True)
    cfr_embeddings = model.encode(eu_annex, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_matrix = F.cosine_similarity(urs_embeddings.unsqueeze(1), cfr_embeddings.unsqueeze(0), dim=-1)

    # Create a mapping defaultdict
    mapping = defaultdict(list)

    # Iterate over the similarity matrix and map each URS point to the most similar CFR point
    for i, urs_id in enumerate(urs_ids):
        max_score_idx = similarity_matrix[i].argmax().item()
        mapping[eu_annex[max_score_idx]].append(urs_id)

    # Format the mapping result
    result = [{"EU Annex": eu, "URS Reference IDs": ", ".join(urs_ids)} for eu, urs_ids in mapping.items()]

    # Return the result as JSON
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6900)
