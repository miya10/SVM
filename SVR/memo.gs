function kudo_notify() {
  var day_num = new Date().getDay()
  if (day_num === 0) {
    return
  }
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Part_time-manager")
  var data = sheet.getDataRange().getValues()
  var today = new Date().getDate()
  var name_list = ['松崎', '米田']
  var message = ''
  for(i=1; i<data.length; i++) {
    var sheet_date = data[i][0].getDate()
    var sheet_name = data[i][1]
    if (sheet_date == today && name_list.indexOf(sheet_name) != -1) {
      Logger.log(sheet_name)
      Logger.log(sheet_date)
      message += sheet_name+'\n'
    }
  }
  Logger.log(today)
  Logger.log(message)
  var kudo_url = 'https://hooks.slack.com/services/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'

  send_slack(kudo_url, 'バイト通知', message, ':calendar:')
}
