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
#ifndef _INC_IMAGEREADER
#define _INC_IMAGEREADER
#include<afx.h>
typedef unsigned short WORD;
typedef unsigned long  DWORD;
/*
*	Data structures for many other image format
*/

typedef struct tagIDFEntry
{
	WORD	iTag;			// tag 
	WORD	iType;
	DWORD	iNumOfValues;	// num of values
	DWORD	iValueOffset;	// value offset
}IDFEntry;

typedef struct tagTiffHdr
{
	WORD	ihWordOrder;
	WORD	ihTiffID;
	DWORD	ihOffset;
}TiffHdr;
typedef struct TagTifHead
{
	TiffHdr HeadTif;
	WORD	nNumOfEntry;
	IDFEntry pIDFEntry[11];
	DWORD	nNextIDF;
	DWORD m_nXResolution1;
	DWORD m_nYResolution1;
	DWORD m_nXResolution2;
	DWORD m_nYResolution2;
}TifHead;
enum {
	iImageWidth = 0x100,
	iImageLegth = 0x101,
	iBitsPerSample = 0x102,
	iCompression = 0x103,
	iPhotometricInterpretation = 0x106,
	iStripOffsets = 0x111,
	iRowsPerStrip = 0x116,
	iStripByteCounts = 0x117,
	iXResolution = 0x11A,
	iYResolution = 0x11B,
	iResolutionUnit = 0x128
};
/*
*	End of definition
*/

class CImageIO	// image io class
{
public:
	enum{
		imfRAW8	= (int)1,
		imfBMP = (int)2,	// IBM/PC bitmap format
		imfTIFF = (int)3,	// general tiff format
		imfRAW16 = (int)4,	// raw 16 bit format
		imfTIFF16 =(int)5,//16 bit Tiff format;
		imfUnknown = (int)0	// unknown format
	};
	enum{
		emErrorOpenFile = (unsigned char)1,
		emErrorCreateFile = (unsigned char)2,
		emErrorWrongType = (unsigned char)3,
		emNoError = (unsigned char)0
	};
	enum{
		tifResUnitInch = 0x02,
		tifResUnitCM = 0x03,
		tifResUnitNoUnit = 0x00
	};

public:
	CString m_szFileName;	// image name
	long m_nXSize;			// real width of images
	long m_nYSize;			// real height of images
	long m_nFrame;			// total number of nFrame available in the frame
	long m_nBitPerPixel;	// Bits per pixel
	int	 m_nImType;			// image type
	long m_nPhotoMetricInterpretation;	// interpret 0 - black or 0 - white
	DWORD m_nXResolution;
	DWORD m_nYResolution;
	WORD  m_nResolutionUnit;

protected:	// image status
	union tagImgStatus{
		long	nReserved;	// force data to be 4-bytes
		struct {
			unsigned short ReadSucceed:1;		// correctly read file in;
			unsigned short WriteSucceed:1;		// correctly write image file;
			unsigned short ErrorMsg:4;
		}flags;
	}ImgStatus;

public:		// operation
	//void SetFileName( CString szFileName );
	//void GetFileName( CString szFileName );

	void SetImageSize( long p_nXSize, long p_nYSize );
	void GetImageSize( long& p_nXSize, long& p_nYSize );

	void ImRead( void *lpImBuf );	// general image reading rountine
	void ImWrite(void *lpImBuf );	// general image writing rountine
	void getImHeadTiff();

protected:	// low level operator
	void ImReadRaw8(void *pImage);		// load data 
	void ImReadRaw8(void **pImage);	// load data

	void ImReadTiff8(unsigned char * pImage);
	void ImReadTiff(void *pImage);
	void ImReadTiff8(unsigned char **pImage);   
	void ImReadTiff16(WORD *pImage);

	
protected:
	
	// low level write operator
	void ImWriteRaw8(void *pImage);
	void ImWriteRaw8(void **pImage);
	
	void ImWriteTiff(void *pImage);
	void ImWriteTiff8(unsigned char *pImage);
	void ImWriteTiff8( unsigned char **pImage);
	void ImWriteTiff16(WORD *pImage);
};

#endif	// _inc_imagereader
