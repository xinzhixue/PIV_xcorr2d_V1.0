#include "FolderFileOp.h"

using namespace std;

/*Examine if a folder exist*/
bool ffop::FolderExist(CString folderName) 
{
	std::string s = CT2A(folderName);
	if (_access(s.c_str(), 0) == -1) {
		//Folder not found
		return false;
	}

	DWORD attr = GetFileAttributes(folderName);
	if (!(attr & FILE_ATTRIBUTE_DIRECTORY)) {
		// Folder invalid
		return false;
	}
	return true;
}

/*Create a folder*/
bool ffop::CreateFolder(CString folderName)
{
	if (CreateDirectory((LPCTSTR)folderName, NULL)) return true;
	else return false;
}

bool ffop::FileExist(CString fileName)
{
	std::string s = CT2A(fileName);
	if (_access(s.c_str(), 0) == -1) {
		//Folder not found
		return false;
	}
	return true;
}