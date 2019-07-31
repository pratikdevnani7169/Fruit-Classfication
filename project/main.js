const electron = require('electron')

const app = electron.app

const BrowserWindow = electron.BrowserWindow

const path = require('path')
const url = require('url')

let mainWindow

function createWindow () {
  
  mainWindow = new BrowserWindow({width: 1000, height: 800})

  
  mainWindow.loadURL('http:localhost:5500/')

  mainWindow.on('closed', function () {
   
    mainWindow = null
  })
}

app.on('ready', createWindow)

app.on('window-all-closed', function () {
  

    app.quit()
})

app.on('activate', function () {
  if (mainWindow === null) {
    createWindow()
  }
})


