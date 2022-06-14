/* File : example.c */

#include<iostream>
#include<fstream>
#include<string>
#include<cctype>
#include <bits/stdc++.h> 
using namespace std;
string keywords[] = {"int","float","return","double","cout","using","namespace","std"};
string brackets[] = {"}","{","(",")","[","]"};
string operators[] = {"=","+","-","*","/","<<"};
string symbols[] = {";"};

bool isContained(string query,string arr[],int s)
{
 for(int i=0;i<s;i++)
 {
     if(query == arr[i])
         return true;
 }
 return false;
}
bool isConstant(string query)
{
 for(int i=0;i<query.size();i++)
 {
     if(!isdigit(query[i]))
         return false;
 }
 return true;
}
bool isIdentifier(string query)
{
 if(isdigit(query[0]))
     return false;
 for(int i=1;i<query.size();i++)
 {
     if(!isalnum(query[i]))
         return false;
 }
 return true;
}

char* etiq(char *fil){
    char *out;
    int filas=40; // En el TG voy a estar procesando 160 lineas
    out = new char[filas];

    fstream file;
    string line,word="";
    bool flag=false;
    string filename;
    filename=fil;
    file.open(filename,ios::in);
    //cout<<"Token\t\t\tClass"<<endl;
    //cout<<"-----\t\t\t-----"<<endl;
    //string **out;
    int count=0;

    while(getline(file,line))
    {
        //cout<<line<<"-"<<endl;
        line.push_back('\0'); //Append line
        if(line[0] == '#') //Pienso adicionar caracteres especiales
            cout<<line<<"\tPreprocessor Directive"<<endl;
        else
        {
            if(line.find("()") != string::npos)//find no devuelve falso, hay que comparar
            {
                flag = true;
            }
            for(int i=0;i<line.size();i++)
            {
                if(line[i] == ' ' || line[i] == '\0' || (flag && line[i]=='('))
                {
                    count=count+1;
                    if(flag && line[i]=='(')
                    {
                        out[count]='B';
                    }
                    else if(isContained(word,keywords,8))
                        out[count]='K';
                    else if(isContained(word,brackets,6))
                        out[count]='B';
                    else if(isContained(word,operators,6))
                        out[count]='O';
                    else if(isContained(word,symbols,1))
                        out[count]='S';
                    else if(isConstant(word))
                        out[count]='C';
                    else if(isIdentifier(word))
                        out[count]='I';
                    else
                        out[count]='U';
                    word.clear(); 
                }else{
                    word.push_back(line[i]);
                }}
        }
        word.clear();
        flag = false;
        //cout<<line;
    }
    file.close();
    return out;
}

int suma(int a, int b){
  return a+b;
}