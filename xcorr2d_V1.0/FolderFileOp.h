#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "io.h"
#include <list>
#include <afx.h>
#include "stdafx.h"

using namespace std;

class ffop
{
public:
	bool ffop::FolderExist(CString folderName);
	bool ffop::CreateFolder(CString folderName);
	bool ffop::FileExist(CString fileName);
};