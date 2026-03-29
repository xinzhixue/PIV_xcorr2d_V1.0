/***************************************************************************
*	Program	: CImageIO
*	
*	Description	:
*		General class for image I/O operations
*	
*	Author	:	Jian Sheng
*	
*	Date	:	12/21/2001
*	
*	Version	:	1.0.0.0
*	Modification:
*	
*	Person		Description								Date		Version
*	------------------------------------------------------------------------
*	JS			Initial put								12/21/01	1.0.0.0
*	------------------------------------------------------------------------
*	
*
***************************************************************************/

#include "stdafx.h"			// for microsoft
#include "ImageIOPkg.h"		// image i/o handling routine
#include<afx.h>

//void CImageIO::SetFileName(CString szFileName)
//{
//	szFileName.
	//strcpy( m_szFileName, szFileName );
//	return;
//}

//void CImageIO::GetFileName(CString szFileName )
//{
//	strcpy( szFileName, m_szFileName );
//	return;
//}

void CImageIO::SetImageSize( long p_nXSize, long p_nYSize )
{
	m_nXSize = p_nXSize;
	m_nYSize = p_nYSize;

	return;
}

void CImageIO::GetImageSize( long& p_nXSize, long& p_nYSize )
{
	p_nXSize = m_nXSize;
	p_nYSize = m_nYSize;

	return;
}

void CImageIO::ImRead( void *lpImBuf )
{
	switch(m_nImType){
		case imfRAW8:
			ImReadRaw8( lpImBuf );
			break;
		case imfTIFF:
			ImReadTiff( lpImBuf );
			break;
		default:
			break;
	}

	return;
}

void CImageIO::ImWrite( void *lpImBuf )
{
	switch(m_nImType){
		case imfRAW8:
			ImWriteRaw8( lpImBuf );
			break;
		case imfTIFF:
			ImWriteTiff( lpImBuf );
		default:
			break;
	}

	return;
}

void CImageIO::ImReadRaw8(void *pImage)
{
	CFile pFile;
	if ( !pFile.Open(m_szFileName, CFile::modeRead,NULL) ){
		ImgStatus.flags.ReadSucceed = 0;	// not correctly read
		ImgStatus.flags.ErrorMsg = 1;		// error opening file
		return;
	}
	
	pFile.Read( pImage, (unsigned long)m_nXSize*(unsigned long)m_nYSize );
	pFile.Close();
	ImgStatus.flags.ReadSucceed = 1;
	ImgStatus.flags.ErrorMsg = 0;
	return;
}

void CImageIO::ImReadRaw8(void **pImage)
{
	CFile pFile;
	int i;

	if ( !pFile.Open( m_szFileName, CFile::modeRead ) ){
		ImgStatus.flags.ReadSucceed = 0;	// not correctly read
		ImgStatus.flags.ErrorMsg = 1;		// error opening file
		return;
	}
	
	for(i=0;i<m_nYSize;i++){
		pFile.Read( pImage[i], m_nXSize );
	}
	pFile.Close();

	ImgStatus.flags.ReadSucceed = 1;
	ImgStatus.flags.ErrorMsg = 0;

	return;
}

void CImageIO::getImHeadTiff()
{
	CFile pFile;
	int i, j;

	if (!pFile.Open(m_szFileName, CFile::modeRead)) {
		ImgStatus.flags.ReadSucceed = 0;	// not correctly read
		ImgStatus.flags.ErrorMsg = 1;		// error opening file
		return;
	}

	// read tif file header;
	WORD	nWordTmp;
	DWORD	nDWordTmp;

	WORD	nWordOrder;
	WORD	nTiffIdn;
	DWORD	nIDFOffset;
	WORD	nIDFEntryNum;
	IDFEntry pIDFEntry;
	long	nNumOfStrips = 1;				// num of strips
	DWORD	*pStripOffset = NULL;			// strip offsets
	DWORD	*pStripByteCount = NULL;
	DWORD	posFile;
	DWORD	nRowsPerStrip;

	DWORD	nXResolution;
	DWORD	nYResolution;
	DWORD	nNextIDF;

	pFile.Read(&nWordOrder, 2);
	pFile.Read(&nTiffIdn, 2);
	// check validity of the file
	if (nTiffIdn != 42) {
		ImgStatus.flags.ReadSucceed = 0;
		ImgStatus.flags.ErrorMsg = emErrorWrongType;
		pFile.Close();
		return;
	}

	pFile.Read(&nIDFOffset, 4);				// byte offset for the first IDF entry

	do {		// loop to read all IDFs
		pFile.Seek(nIDFOffset, CFile::begin);		// offset the IDF
		pFile.Read(&nIDFEntryNum, 2);			// read in Number of IDF Entries
		for (i = 0; i < nIDFEntryNum; i++) {
			pFile.Read(&pIDFEntry, sizeof(pIDFEntry));
			// implementation of Tiff 6.0 baseline grayscale image
			switch (pIDFEntry.iTag) {
			case iImageWidth:
				memcpy(&m_nXSize, &pIDFEntry.iValueOffset, 4);
				break;
			case iImageLegth:
				memcpy(&m_nYSize, &pIDFEntry.iValueOffset, 4);
				break;
			case iBitsPerSample:
				memcpy(&m_nBitPerPixel, &pIDFEntry.iValueOffset, 4);
				break;
			case iCompression:
				break;
			case iPhotometricInterpretation: // 0 - WhiteIsZero or 1 - BlackIsZero
				memcpy(&m_nPhotoMetricInterpretation, &pIDFEntry.iValueOffset, 4);
				break;

			case iStripOffsets:
				nNumOfStrips = pIDFEntry.iNumOfValues;	// number of values
				pStripOffset = new DWORD[nNumOfStrips];
				posFile = pFile.GetPosition();
				if (nNumOfStrips > 1) {
					pFile.Seek(pIDFEntry.iValueOffset, CFile::begin);
					for (j = 0; j < nNumOfStrips; j++) {
						if (pIDFEntry.iType == 0x03) {	// short type
							pFile.Read(&nWordTmp, 2);
							pStripOffset[j] = nWordTmp;
						}
						if (pIDFEntry.iType == 0x04) { // long type
							pFile.Read(&nDWordTmp, 4);
							pStripOffset[j] = nDWordTmp;
						}
					}
					pFile.Seek(posFile, CFile::begin);
				}
				else {
					pStripOffset[0] = pIDFEntry.iValueOffset;
				}
				break;

			case iRowsPerStrip:
				nRowsPerStrip = pIDFEntry.iValueOffset;
				break;

			case iStripByteCounts:
				nNumOfStrips = pIDFEntry.iNumOfValues;	// number of values
				pStripByteCount = new DWORD[nNumOfStrips];
				posFile = pFile.GetPosition();
				if (nNumOfStrips > 1) {
					pFile.Seek(pIDFEntry.iValueOffset, CFile::begin);
					for (j = 0; j < nNumOfStrips; j++) {
						if (pIDFEntry.iType == 0x03) {	// short type
							pFile.Read(&nWordTmp, 2);
							pStripByteCount[j] = nWordTmp;
						}
						if (pIDFEntry.iType == 0x04) { // long type
							pFile.Read(&nDWordTmp, 4);
							pStripByteCount[j] = nDWordTmp;
						}
					}
					pFile.Seek(posFile, CFile::begin);
				}
				else {
					pStripByteCount[0] = pIDFEntry.iValueOffset;
				}
				break;

			case iXResolution:
				memcpy(&m_nXResolution, &pIDFEntry.iValueOffset, 4);
				break;

			case iYResolution:
				memcpy(&m_nYResolution, &pIDFEntry.iValueOffset, 4);
				break;
			case iResolutionUnit:
				memcpy(&m_nResolutionUnit, &pIDFEntry.iValueOffset, 2);
				break;
			default:
				break;
			}
		}
		pFile.Read(&nNextIDF, 4);
		nIDFOffset = nNextIDF;
	} while (nNextIDF != 0);
	delete[]pStripOffset;
	delete[]pStripByteCount;
	pFile.Close();
}
void CImageIO::ImReadTiff(void *pImage)
{
	if (m_nBitPerPixel == 8)
	{
		unsigned char* lpBuf = (unsigned char*)pImage;
		ImReadTiff8(lpBuf);
		lpBuf = NULL;
	}
	if (m_nBitPerPixel == 16)
	{

		WORD* lpBuf2 = (WORD*)pImage;
		ImReadTiff16(lpBuf2);
		lpBuf2 = NULL;
		
	}

	
}


void CImageIO::ImReadTiff8(unsigned char *pImage)
{
	CFile pFile;
	int i, j;

	if (!pFile.Open(m_szFileName, CFile::modeRead)) {
		ImgStatus.flags.ReadSucceed = 0;	// not correctly read
		ImgStatus.flags.ErrorMsg = 1;		// error opening file
		return;
	}

	// read tif file header;
	WORD	nWordTmp;
	DWORD	nDWordTmp;

	WORD	nWordOrder;
	WORD	nTiffIdn;
	DWORD	nIDFOffset;
	WORD	nIDFEntryNum;
	IDFEntry pIDFEntry;
	long	nNumOfStrips = 1;				// num of strips
	DWORD	*pStripOffset = NULL;			// strip offsets
	DWORD	*pStripByteCount = NULL;
	DWORD	posFile;
	DWORD	nRowsPerStrip;

	DWORD	nXResolution;
	DWORD	nYResolution;
	DWORD	nNextIDF;

	pFile.Read(&nWordOrder, 2);
	pFile.Read(&nTiffIdn, 2);
	// check validity of the file
	if (nTiffIdn != 42) {
		ImgStatus.flags.ReadSucceed = 0;
		ImgStatus.flags.ErrorMsg = emErrorWrongType;
		pFile.Close();
		return;
	}

	pFile.Read(&nIDFOffset, 4);				// byte offset for the first IDF entry

	do {		// loop to read all IDFs
		pFile.Seek(nIDFOffset, CFile::begin);		// offset the IDF
		pFile.Read(&nIDFEntryNum, 2);			// read in Number of IDF Entries
		for (i = 0; i < nIDFEntryNum; i++) {
			pFile.Read(&pIDFEntry, sizeof(pIDFEntry));
			// implementation of Tiff 6.0 baseline grayscale image
			switch (pIDFEntry.iTag) {
			case iImageWidth:
				memcpy(&m_nXSize, &pIDFEntry.iValueOffset, 4);
				break;
			case iImageLegth:
				memcpy(&m_nYSize, &pIDFEntry.iValueOffset, 4);
				break;
			case iBitsPerSample:
				memcpy(&m_nBitPerPixel, &pIDFEntry.iValueOffset, 4);
				break;
			case iCompression:
				break;
			case iPhotometricInterpretation: // 0 - WhiteIsZero or 1 - BlackIsZero
				memcpy(&m_nPhotoMetricInterpretation, &pIDFEntry.iValueOffset, 4);
				break;

			case iStripOffsets:
				nNumOfStrips = pIDFEntry.iNumOfValues;	// number of values
				pStripOffset = new DWORD[nNumOfStrips];
				posFile = pFile.GetPosition();
				if (nNumOfStrips > 1) {
					pFile.Seek(pIDFEntry.iValueOffset, CFile::begin);
					for (j = 0; j < nNumOfStrips; j++) {
						if (pIDFEntry.iType == 0x03) {	// short type
							pFile.Read(&nWordTmp, 2);
							pStripOffset[j] = nWordTmp;
						}
						if (pIDFEntry.iType == 0x04) { // long type
							pFile.Read(&nDWordTmp, 4);
							pStripOffset[j] = nDWordTmp;
						}
					}
					pFile.Seek(posFile, CFile::begin);
				}
				else {
					pStripOffset[0] = pIDFEntry.iValueOffset;
				}
				break;

			case iRowsPerStrip:
				nRowsPerStrip = pIDFEntry.iValueOffset;
				break;

			case iStripByteCounts:
				nNumOfStrips = pIDFEntry.iNumOfValues;	// number of values
				pStripByteCount = new DWORD[nNumOfStrips];
				posFile = pFile.GetPosition();
				if (nNumOfStrips > 1) {
					pFile.Seek(pIDFEntry.iValueOffset, CFile::begin);
					for (j = 0; j < nNumOfStrips; j++) {
						if (pIDFEntry.iType == 0x03) {	// short type
							pFile.Read(&nWordTmp, 2);
							pStripByteCount[j] = nWordTmp;
						}
						if (pIDFEntry.iType == 0x04) { // long type
							pFile.Read(&nDWordTmp, 4);
							pStripByteCount[j] = nDWordTmp;
						}
					}
					pFile.Seek(posFile, CFile::begin);
				}
				else {
					pStripByteCount[0] = pIDFEntry.iValueOffset;
				}
				break;

			case iXResolution:
				memcpy(&m_nXResolution, &pIDFEntry.iValueOffset, 4);
				break;

			case iYResolution:
				memcpy(&m_nYResolution, &pIDFEntry.iValueOffset, 4);
				break;
			case iResolutionUnit:
				memcpy(&m_nResolutionUnit, &pIDFEntry.iValueOffset, 2);
				break;
			default:
				break;
			}
		}
		pFile.Read(&nNextIDF, 4);
		nIDFOffset = nNextIDF;
	} while (nNextIDF != 0);

	// reading grayscale image
	
	
		unsigned char *lpBufTmp = (unsigned char*)pImage;
		for (i = 0; i < nNumOfStrips; i++) {
			pFile.Seek(pStripOffset[i], CFile::begin);		// find the strip
			pFile.Read(lpBufTmp, pStripByteCount[i]);
			lpBufTmp += pStripByteCount[i];
		}
		lpBufTmp = NULL;
	

	// end of read-in images
	if ( pStripOffset != NULL ) delete []pStripOffset;
	if ( pStripByteCount != NULL ) delete []pStripByteCount;

	return;
}
void CImageIO::ImReadTiff16(WORD *pImage)
{
	CFile pFile;
	int i, j;

	if (!pFile.Open(m_szFileName, CFile::modeRead)) {
		ImgStatus.flags.ReadSucceed = 0;	// not correctly read
		ImgStatus.flags.ErrorMsg = 1;		// error opening file
		return;
	}

	// read tif file header;
	WORD	nWordTmp;
	DWORD	nDWordTmp;

	WORD	nWordOrder;
	WORD	nTiffIdn;
	DWORD	nIDFOffset;
	WORD	nIDFEntryNum;
	IDFEntry pIDFEntry;
	long	nNumOfStrips = 1;				// num of strips
	DWORD	*pStripOffset = NULL;			// strip offsets
	DWORD	*pStripByteCount = NULL;
	DWORD	posFile;
	DWORD	nRowsPerStrip;

	DWORD	nXResolution;
	DWORD	nYResolution;
	DWORD	nNextIDF;

	pFile.Read(&nWordOrder, 2);
	pFile.Read(&nTiffIdn, 2);
	// check validity of the file
	if (nTiffIdn != 42) {
		ImgStatus.flags.ReadSucceed = 0;
		ImgStatus.flags.ErrorMsg = emErrorWrongType;
		pFile.Close();
		return;
	}

	pFile.Read(&nIDFOffset, 4);				// byte offset for the first IDF entry

	do {		// loop to read all IDFs
		pFile.Seek(nIDFOffset, CFile::begin);		// offset the IDF
		pFile.Read(&nIDFEntryNum, 2);			// read in Number of IDF Entries
		for (i = 0; i < nIDFEntryNum; i++) {
			pFile.Read(&pIDFEntry, sizeof(pIDFEntry));
			// implementation of Tiff 6.0 baseline grayscale image
			switch (pIDFEntry.iTag) {
			case iImageWidth:
				memcpy(&m_nXSize, &pIDFEntry.iValueOffset, 4);
				break;
			case iImageLegth:
				memcpy(&m_nYSize, &pIDFEntry.iValueOffset, 4);
				break;
			case iBitsPerSample:
				memcpy(&m_nBitPerPixel, &pIDFEntry.iValueOffset, 4);
				break;
			case iCompression:
				break;
			case iPhotometricInterpretation: // 0 - WhiteIsZero or 1 - BlackIsZero
				memcpy(&m_nPhotoMetricInterpretation, &pIDFEntry.iValueOffset, 4);
				break;

			case iStripOffsets:
				nNumOfStrips = pIDFEntry.iNumOfValues;	// number of values
				pStripOffset = new DWORD[nNumOfStrips];
				posFile = pFile.GetPosition();
				if (nNumOfStrips > 1) {
					pFile.Seek(pIDFEntry.iValueOffset, CFile::begin);
					for (j = 0; j < nNumOfStrips; j++) {
						if (pIDFEntry.iType == 0x03) {	// short type
							pFile.Read(&nWordTmp, 2);
							pStripOffset[j] = nWordTmp;
						}
						if (pIDFEntry.iType == 0x04) { // long type
							pFile.Read(&nDWordTmp, 4);
							pStripOffset[j] = nDWordTmp;
						}
					}
					pFile.Seek(posFile, CFile::begin);
				}
				else {
					pStripOffset[0] = pIDFEntry.iValueOffset;
				}
				break;

			case iRowsPerStrip:
				nRowsPerStrip = pIDFEntry.iValueOffset;
				break;

			case iStripByteCounts:
				nNumOfStrips = pIDFEntry.iNumOfValues;	// number of values
				pStripByteCount = new DWORD[nNumOfStrips];
				posFile = pFile.GetPosition();
				if (nNumOfStrips > 1) {
					pFile.Seek(pIDFEntry.iValueOffset, CFile::begin);
					for (j = 0; j < nNumOfStrips; j++) {
						if (pIDFEntry.iType == 0x03) {	// short type
							pFile.Read(&nWordTmp, 2);
							pStripByteCount[j] = nWordTmp;
						}
						if (pIDFEntry.iType == 0x04) { // long type
							pFile.Read(&nDWordTmp, 4);
							pStripByteCount[j] = nDWordTmp;
						}
					}
					pFile.Seek(posFile, CFile::begin);
				}
				else {
					pStripByteCount[0] = pIDFEntry.iValueOffset;
				}
				break;

			case iXResolution:
				memcpy(&m_nXResolution, &pIDFEntry.iValueOffset, 4);
				break;

			case iYResolution:
				memcpy(&m_nYResolution, &pIDFEntry.iValueOffset, 4);
				break;
			case iResolutionUnit:
				memcpy(&m_nResolutionUnit, &pIDFEntry.iValueOffset, 2);
				break;
			default:
				break;
			}
		}
		pFile.Read(&nNextIDF, 4);
		nIDFOffset = nNextIDF;
	} while (nNextIDF != 0);

	// reading grayscale image

	
	WORD *lpBufTmp = (WORD*)pImage;
	for (i = 0; i < nNumOfStrips; i++) {
		pFile.Seek(pStripOffset[i], CFile::begin);		// find the strip
		pFile.Read(lpBufTmp, pStripByteCount[i]);
		lpBufTmp += pStripByteCount[i]/(m_nBitPerPixel/8);
	}
	lpBufTmp = NULL;
   // end of read-in images
	if (pStripOffset != NULL) delete[]pStripOffset;
	if (pStripByteCount != NULL) delete[]pStripByteCount;

	return;
}

void CImageIO::ImWriteRaw8(void *pImage)
{
	CFile pFile;

	if ( !pFile.Open(m_szFileName, CFile::modeCreate|CFile::modeWrite ) ){
		ImgStatus.flags.WriteSucceed = 0;	// not correctly read
		ImgStatus.flags.ErrorMsg = emErrorCreateFile;		// error opening file
		return;
	}
	
	pFile.Write( pImage, (unsigned long)m_nXSize*m_nYSize );

	pFile.Close();

	ImgStatus.flags.WriteSucceed = 1;
	ImgStatus.flags.ErrorMsg = 0;

	return;
}

void CImageIO::ImWriteRaw8(void **pImage)
{
	CFile pFile;
	int i;

	if (!pFile.Open(m_szFileName, CFile::modeCreate | CFile::modeWrite)) {
		ImgStatus.flags.WriteSucceed = 0;					// not correctly read
		ImgStatus.flags.ErrorMsg = emErrorCreateFile;		// error opening file
		return;
	}

	for (i = 0; i < m_nYSize; i++) {
		pFile.Write(pImage[i], m_nXSize);
	}
	pFile.Close();

	ImgStatus.flags.WriteSucceed = 1;
	ImgStatus.flags.ErrorMsg = 0;

	return;
}
void CImageIO::ImWriteTiff(void * pImage)
{
	if (m_nBitPerPixel == 8)
	{
		unsigned char *lpBufTmp = (unsigned char*)pImage;
		ImWriteTiff8(lpBufTmp);
		lpBufTmp = NULL;
	}
		
	if(m_nBitPerPixel==16)
	{
		WORD *lpBufTmp = (WORD*)pImage;
		ImWriteTiff16(lpBufTmp);
		lpBufTmp = NULL;
	
	}
	
	return;
}


	void CImageIO::ImWriteTiff8( unsigned char *pImage)
{
	CFile pFile;
	WORD	nNumOfEntry;
	DWORD	nNextIDF = 0;
	IDFEntry pIDFEntry[11];
	TiffHdr	 pTiffHdr;

	if ( !pFile.Open( m_szFileName, CFile::modeCreate|CFile::modeWrite ) ){
		ImgStatus.flags.WriteSucceed = 0;					// not correctly read
		ImgStatus.flags.ErrorMsg = emErrorCreateFile;		// error opening file
		return;
	}
	pTiffHdr.ihWordOrder = 0x4949;
	pTiffHdr.ihTiffID = 42;
	pTiffHdr.ihOffset = sizeof(pTiffHdr);

	nNumOfEntry = 11;
	pIDFEntry[0].iTag = iImageWidth;		// ImageWidth
	pIDFEntry[0].iType = 4;					// long
	pIDFEntry[0].iNumOfValues = 1;
	pIDFEntry[0].iValueOffset = m_nXSize;

	pIDFEntry[1].iTag = iImageLegth;		// ImageHeight
	pIDFEntry[1].iType = 4;					// long
	pIDFEntry[1].iNumOfValues = 1;
	pIDFEntry[1].iValueOffset = m_nYSize;

	m_nBitPerPixel = 8;
	pIDFEntry[2].iTag = iBitsPerSample;		// iBitsPerSample
	pIDFEntry[2].iType = 3;					// long
	pIDFEntry[2].iNumOfValues = 1;
	pIDFEntry[2].iValueOffset = m_nBitPerPixel;
	
	pIDFEntry[3].iTag = iCompression;		// iCompression
	pIDFEntry[3].iType = 3;					// long
	pIDFEntry[3].iNumOfValues = 1;
	pIDFEntry[3].iValueOffset = 1;

	m_nPhotoMetricInterpretation = 1;
	pIDFEntry[4].iTag = iPhotometricInterpretation;		// iPhotometricInterpretation
	pIDFEntry[4].iType = 3;					// long
	pIDFEntry[4].iNumOfValues = 1;
	pIDFEntry[4].iValueOffset = m_nPhotoMetricInterpretation;

	pIDFEntry[5].iTag = iStripOffsets;		// iStripOffsets
	pIDFEntry[5].iType = 4;					// long
	pIDFEntry[5].iNumOfValues = 1;
	pIDFEntry[5].iValueOffset = sizeof(pTiffHdr)+sizeof(IDFEntry)*11+sizeof(nNumOfEntry)+4+4*sizeof(long);

	pIDFEntry[6].iTag = iRowsPerStrip;		// iRowsPerStrip
	pIDFEntry[6].iType = 4;					// long
	pIDFEntry[6].iNumOfValues = 1;
	pIDFEntry[6].iValueOffset = m_nYSize;
	
	pIDFEntry[7].iTag = iStripByteCounts;		// iRowsPerStrip
	pIDFEntry[7].iType = 4;					// long
	pIDFEntry[7].iNumOfValues = 1;
	pIDFEntry[7].iValueOffset = (unsigned long)m_nYSize*m_nXSize;

	pIDFEntry[8].iTag = iXResolution;		// iRowsPerStrip
	pIDFEntry[8].iType = 5;					// long
	pIDFEntry[8].iNumOfValues = 1;
	pIDFEntry[8].iValueOffset = sizeof(pTiffHdr)+sizeof(IDFEntry)*11+sizeof(nNumOfEntry)+4;

	pIDFEntry[9].iTag = iYResolution;		// iRowsPerStrip
	pIDFEntry[9].iType = 5;					// long
	pIDFEntry[9].iNumOfValues = 1;
	pIDFEntry[9].iValueOffset = sizeof(pTiffHdr)+sizeof(IDFEntry)*11+sizeof(nNumOfEntry)+4+2*sizeof(long);

	pIDFEntry[10].iTag = iResolutionUnit;		// iRowsPerStrip
	pIDFEntry[10].iType = 3;					// short
	pIDFEntry[10].iNumOfValues = 1;
	pIDFEntry[10].iValueOffset = tifResUnitInch;

	m_nXResolution = 720000;
	m_nYResolution = 10000;

	pFile.Write( &pTiffHdr, sizeof(pTiffHdr) );
	pFile.Write( &nNumOfEntry, sizeof( nNumOfEntry ) );
	pFile.Write( pIDFEntry, sizeof(IDFEntry)*11 );
	pFile.Write( &nNextIDF, 4 );
	pFile.Write( &m_nXResolution, 4 );
	pFile.Write( &m_nYResolution, 4 );
	pFile.Write( &m_nXResolution, 4 );
	pFile.Write( &m_nYResolution, 4 );

	pFile.Write( pImage, (unsigned long)m_nXSize*m_nYSize );

	pFile.Close();
	return;
}


void CImageIO::ImWriteTiff16(WORD *pImage)
{
	CFile pFile;
	WORD	nNumOfEntry;
	DWORD	nNextIDF = 0;
	IDFEntry pIDFEntry[11];
	TiffHdr	 pTiffHdr;

	if (!pFile.Open(m_szFileName, CFile::modeCreate | CFile::modeWrite)) {
		ImgStatus.flags.WriteSucceed = 0;					// not correctly read
		ImgStatus.flags.ErrorMsg = emErrorCreateFile;		// error opening file
		return;
	}
	pTiffHdr.ihWordOrder = 0x4949;
	pTiffHdr.ihTiffID = 42;
	pTiffHdr.ihOffset = sizeof(pTiffHdr);

	nNumOfEntry = 11;
	pIDFEntry[0].iTag = iImageWidth;		// ImageWidth
	pIDFEntry[0].iType = 4;					// long
	pIDFEntry[0].iNumOfValues = 1;
	pIDFEntry[0].iValueOffset = m_nXSize;

	pIDFEntry[1].iTag = iImageLegth;		// ImageHeight
	pIDFEntry[1].iType = 4;					// long
	pIDFEntry[1].iNumOfValues = 1;
	pIDFEntry[1].iValueOffset = m_nYSize;

	m_nBitPerPixel = 16;
	pIDFEntry[2].iTag = iBitsPerSample;		// iBitsPerSample
	pIDFEntry[2].iType = 3;					// long
	pIDFEntry[2].iNumOfValues = 1;
	pIDFEntry[2].iValueOffset = m_nBitPerPixel;

	pIDFEntry[3].iTag = iCompression;		// iCompression
	pIDFEntry[3].iType = 3;					// long
	pIDFEntry[3].iNumOfValues = 1;
	pIDFEntry[3].iValueOffset = 1;

	m_nPhotoMetricInterpretation = 1;
	pIDFEntry[4].iTag = iPhotometricInterpretation;		// iPhotometricInterpretation
	pIDFEntry[4].iType = 3;					// long
	pIDFEntry[4].iNumOfValues = 1;
	pIDFEntry[4].iValueOffset = m_nPhotoMetricInterpretation;

	pIDFEntry[5].iTag = iStripOffsets;		// iStripOffsets
	pIDFEntry[5].iType = 4;					// long
	pIDFEntry[5].iNumOfValues = 1;
	pIDFEntry[5].iValueOffset = sizeof(pTiffHdr) + sizeof(IDFEntry) * 11 + sizeof(nNumOfEntry) + 4 + 4 * sizeof(long);

	pIDFEntry[6].iTag = iRowsPerStrip;		// iRowsPerStrip
	pIDFEntry[6].iType = 4;					// long
	pIDFEntry[6].iNumOfValues = 1;
	pIDFEntry[6].iValueOffset = m_nYSize;

	pIDFEntry[7].iTag = iStripByteCounts;		// iRowsPerStrip
	pIDFEntry[7].iType = 4;					// long
	pIDFEntry[7].iNumOfValues = 1;
	pIDFEntry[7].iValueOffset = (unsigned long)m_nYSize*m_nXSize;

	pIDFEntry[8].iTag = iXResolution;		// iRowsPerStrip
	pIDFEntry[8].iType = 5;					// long
	pIDFEntry[8].iNumOfValues = 1;
	pIDFEntry[8].iValueOffset = sizeof(pTiffHdr) + sizeof(IDFEntry) * 11 + sizeof(nNumOfEntry) + 4;

	pIDFEntry[9].iTag = iYResolution;		// iRowsPerStrip
	pIDFEntry[9].iType = 5;					// long
	pIDFEntry[9].iNumOfValues = 1;
	pIDFEntry[9].iValueOffset = sizeof(pTiffHdr) + sizeof(IDFEntry) * 11 + sizeof(nNumOfEntry) + 4 + 2 * sizeof(long);

	pIDFEntry[10].iTag = iResolutionUnit;		// iRowsPerStrip
	pIDFEntry[10].iType = 3;					// short
	pIDFEntry[10].iNumOfValues = 1;
	pIDFEntry[10].iValueOffset = tifResUnitInch;

	m_nXResolution = 720000;
	m_nYResolution = 10000;

	pFile.Write(&pTiffHdr, sizeof(pTiffHdr));
	pFile.Write(&nNumOfEntry, sizeof(nNumOfEntry));
	pFile.Write(pIDFEntry, sizeof(IDFEntry) * 11);
	pFile.Write(&nNextIDF, 4);
	pFile.Write(&m_nXResolution, 4);
	pFile.Write(&m_nYResolution, 4);
	pFile.Write(&m_nXResolution, 4);
	pFile.Write(&m_nYResolution, 4);

	pFile.Write(pImage, (unsigned long)m_nXSize*m_nYSize*2);
    pFile.Close();
	return;
}
