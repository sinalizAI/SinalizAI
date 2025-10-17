// Test SendGrid send using @sendgrid/mail
const sgMail = require('@sendgrid/mail')
const fs = require('fs')
const path = require('path')

let apiKey = process.env.SENDGRID_API_KEY

if (!apiKey) {
  // try to read functions/.runtimeconfig.json for sendgrid.api_key
  try {
    const rcPath = path.join(__dirname, '.runtimeconfig.json')
    if (fs.existsSync(rcPath)) {
      const rc = JSON.parse(fs.readFileSync(rcPath, 'utf8'))
      if (rc && rc.sendgrid && rc.sendgrid.api_key) {
        apiKey = rc.sendgrid.api_key
        console.log('Using sendgrid.api_key from .runtimeconfig.json')
      }
    }
  } catch (err) {
    // ignore parse errors
  }
}

if (!apiKey) {
  console.error('ERROR: SENDGRID_API_KEY not set in environment nor in .runtimeconfig.json')
  process.exit(2)
}

sgMail.setApiKey(apiKey)

// Optional: set data residency to EU when requested
let dataResidency = process.env.SENDGRID_DATA_RESIDENCY
if (!dataResidency) {
  try {
    const rcPath = require('path').join(__dirname, '.runtimeconfig.json')
    if (require('fs').existsSync(rcPath)) {
      const rc = JSON.parse(require('fs').readFileSync(rcPath, 'utf8'))
      if (rc && rc.sendgrid && (rc.sendgrid.dataResidency || rc.sendgrid.data_residency)) {
        dataResidency = rc.sendgrid.dataResidency || rc.sendgrid.data_residency
      }
    }
  } catch (e) {
    // ignore
  }
}

if (dataResidency && String(dataResidency).toLowerCase() === 'eu') {
  if (typeof sgMail.setDataResidency === 'function') {
    sgMail.setDataResidency('eu')
    console.log('SendGrid data residency set to EU')
  } else {
    console.log('sgMail.setDataResidency is not available in this @sendgrid/mail version')
  }
}

const msg = {
  to: process.env.SG_TO || 'teste@exemplo.com', // change to a real recipient for full test
  from: process.env.SG_FROM || '***EMAIL_REMOVED***', // must be a verified sender in SendGrid
  subject: 'Sending with SendGrid is Fun',
  text: 'and easy to do anywhere, even with Node.js',
  html: '<strong>and easy to do anywhere, even with Node.js</strong>',
}
// optional replyTo
if (process.env.SG_REPLY_TO) {
  msg.replyTo = process.env.SG_REPLY_TO
}

sgMail
  .send(msg)
  .then(() => {
    console.log('Email sent')
    process.exit(0)
  })
  .catch((error) => {
    console.error('SendGrid error:')
    if (error.response && error.response.body) {
      console.error(JSON.stringify(error.response.body, null, 2))
    } else {
      console.error(error)
    }
    process.exit(1)
  })
