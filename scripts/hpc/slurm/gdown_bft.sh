OUTDIR=/BS/diffusion-track/nobackup/data/mot

# https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work
# Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
# In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
# Click Authorize APIs and then Exchange authorization code for tokens
# Copy the Access token
ACCESS_TOKEN=$1

mkdir -p ${OUTDIR}/BFT

FILE_ID=1iEebl-2yPjapQByOotoLG_0ud_1q_hZs  # train
FILE_NAME=${OUTDIR}/BFT/train.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media -o $FILE_NAME 

FILE_ID=1fwswUfxxmvcd7GQhveXThVYfK0eR0nbw  # val
FILE_NAME=${OUTDIR}/BFT/val.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media -o $FILE_NAME 

# FILE_ID=1OHXmtxyI_H6uoopyZZRDeNAsTZF9qXkp  # test
# FILE_NAME=${OUTDIR}/BFT/test.zip
# curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media -o $FILE_NAME 
