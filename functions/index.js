const functions = require('firebase-functions');
const admin = require('firebase-admin');
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const sgMail = require('@sendgrid/mail');


(() => {

  let sgKey = process.env.SENDGRID_API_KEY || process.env.SG_API_KEY || '';
  if (!sgKey) {
    try {
      const rc = require('./.runtimeconfig.json');
      if (rc && rc.sendgrid && rc.sendgrid.api_key) sgKey = rc.sendgrid.api_key;
    } catch (e) {

    }
  }
  if (sgKey) {
    try { sgMail.setApiKey(sgKey); } catch (e) { console.warn('Failed to set SendGrid API key:', e && e.message); }
  }
})();

admin.initializeApp();

const app = express();
app.use(cors({ origin: true }));
app.use(express.json());

const TEMPLATE_FEEDBACK = 'd-2b5591d285cb48bbb33ee5c45439f8f8';
const TEMPLATE_VERIFICATION = 'd-643e9026010744e58f392c3e59061ec5';


const isSendGridConfigured = () => {
  if (process.env.SENDGRID_API_KEY || process.env.SG_API_KEY) return true;
  try {
    const rc = require('./.runtimeconfig.json');
    if (rc && rc.sendgrid && rc.sendgrid.api_key) return true;
  } catch (e) {  }
  return false;
  return false;
};


const makeNowAndExpiry = (minutes = 120) => {
  const ms = Date.now();
  const expiryMs = ms + minutes * 60 * 1000;
  try {
    if (admin && admin.firestore && admin.firestore.Timestamp && typeof admin.firestore.Timestamp.now === 'function') {
      return {
        createdAt: admin.firestore.Timestamp.now(),
        expiresAt: admin.firestore.Timestamp.fromMillis(expiryMs),
        _ms: ms
      };
    }
  } catch (e) {

  }
  return { createdAt: ms, expiresAt: expiryMs, _ms: ms };
};


const millisFromFirestoreValue = (val) => {
  if (!val) return null;
  try {
    if (typeof val.toMillis === 'function') return val.toMillis();
  } catch (e) {  }
  if (typeof val === 'number') return val;

  if (val && typeof val._seconds === 'number') return (val._seconds * 1000) + Math.floor((val._nanoseconds || 0) / 1e6);
  return null;
};


app.post('/sendFeedback', async (req, res) => {
  try {
    const { user_email, user_name, subject, message } = req.body;
    if (!user_email || !user_name || !subject || !message) {
      return res.status(400).json({ success: false, message: 'Missing fields' });
    }


    let feedback_email = process.env.FEEDBACK_EMAIL || process.env.SENDGRID_FROM || '';
    if (!feedback_email) {
      try { const rc = require('./.runtimeconfig.json'); if (rc && rc.sendgrid && rc.sendgrid.feedback_email) feedback_email = rc.sendgrid.feedback_email || ''; } catch (e) {  }
    }

    const isEmulator = !!process.env.FIRESTORE_EMULATOR_HOST || !!process.env.FIREBASE_EMULATOR_HUB;
    const hasSG = isSendGridConfigured();


    let sgFrom = process.env.SENDGRID_FROM || '';
    if (!sgFrom) {
      try { const rc = require('./.runtimeconfig.json'); if (rc && rc.sendgrid && rc.sendgrid.from) sgFrom = rc.sendgrid.from || ''; } catch (e) {  }
    }

    const msg = {
      to: feedback_email || user_email,
      from: sgFrom || (feedback_email || user_email),
      templateId: TEMPLATE_FEEDBACK,
      dynamic_template_data: {
        from_name: user_name,
        from_email: user_email,
        subject,
        message
      },
      replyTo: user_email
    };

    try {
      if (!hasSG && !isEmulator) {
        return res.status(500).json({ success: false, message: 'SendGrid não está configurado. Defina sendgrid.api_key e sendgrid.from via `firebase functions:config:set`.' });
      }

      if (hasSG) {

        if (!sgFrom && !isEmulator) {
          return res.status(500).json({ success: false, message: 'sendgrid.from não está configurado. Defina um remetente verificado.' });
        }

        msg.from = sgFrom || msg.from;
        msg.to = feedback_email || msg.to;
        await sgMail.send(msg);
        return res.json({ success: true, via: 'sendgrid' });
      }


      console.warn('SendGrid não configurado; modo emulador - simulando envio de feedback');
      console.log('Simulated feedback message:', msg);
      return res.json({ success: true, simulated: true });
    } catch (err) {

      try {
        const status = err && err.code ? err.code : err && err.response && err.response.status ? err.response.status : 'unknown';
        const respBody = err && err.response && (err.response.body || err.response.data) ? (err.response.body || err.response.data) : null;
        console.error('Feedback send unexpected error:', { message: err && err.message, status, respBody });
        const body = respBody || err && err.message || 'Unknown error';
        return res.status(500).json({ success: false, message: body });
      } catch (logErr) {
        console.error('Error while logging send error', logErr);
        return res.status(500).json({ success: false, message: err && err.message });
      }
    }
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

exports.api = functions.https.onRequest(app);




const callIdentityToolkit = async (path, payload) => {

  let apiKey = '';

  apiKey = process.env.FIREBASE_API_KEY || process.env.FIREBASE_WEB_API_KEY || '';
  if (!apiKey) {
    try {


      const rc = require('./.runtimeconfig.json');
      if (rc && rc.firebase && rc.firebase.api_key) apiKey = rc.firebase.api_key;
    } catch (e) {

    }
  }

  if (!apiKey) throw new Error('Firebase API key not configured in functions.config().firebase.api_key or .runtimeconfig.json or env FIREBASE_API_KEY');
  const url = `https://identitytoolkit.googleapis.com/v1/accounts:${path}?key=${apiKey}`;
  return axios.post(url, payload, { headers: { 'Content-Type': 'application/json' } });
};


app.post('/register', async (req, res) => {
  try {
    const { email, password, displayName, accepted_terms_of_service, accepted_privacy_policy } = req.body || {};
    if (!email || !password || !displayName) return res.status(400).json({ success: false, message: 'Missing fields' });


    if (displayName.trim().length < 3) return res.status(400).json({ success: false, message: 'Nome muito curto' });
    if (password.length < 6) return res.status(400).json({ success: false, message: 'Senha deve ter ao menos 6 caracteres' });


    let userRecord;
    try {
      userRecord = await admin.auth().createUser({ email, password, displayName });
    } catch (e) {

      if (e && e.code === 'auth/email-already-exists') {
        return res.status(400).json({ success: false, message: 'Usuário já existe. Se você não lembra a senha, use o campo "Esqueci minha senha".' });
      }
      throw e;
    }



    let createdAtValue;
    try {
      createdAtValue = admin.firestore.FieldValue.serverTimestamp();
    } catch (e) {
      createdAtValue = Date.now();
    }
    await admin.firestore().collection('users').doc(userRecord.uid).set({
      email,
      displayName,
      createdAt: createdAtValue
    });


    try {
      if (accepted_terms_of_service || accepted_privacy_policy) {
        const acceptedAt = (createdAtValue && createdAtValue.toMillis) ? createdAtValue : admin.firestore.Timestamp ? admin.firestore.Timestamp.now() : Date.now();
        await admin.firestore().collection('legal_acceptances').doc(userRecord.uid).set({
          accepted_terms_of_service: !!accepted_terms_of_service,
          accepted_privacy_policy: !!accepted_privacy_policy,
          accepted_at: acceptedAt,
          user_id: userRecord.uid
        });
      }
    } catch (e) {
      console.warn('Could not save legal acceptance server-side:', e && e.message ? e.message : e);

    }


    try {
      const customToken = await admin.auth().createCustomToken(userRecord.uid);

      return res.json({ success: true, data: { uid: userRecord.uid, customToken }, message: 'Usuário criado com sucesso. Use o token para efetuar login.' });
    } catch (err) {
      console.error('Error creating custom token', err && err.message ? err.message : err);
      return res.status(500).json({ success: false, message: 'Erro interno ao criar credenciais. Tente novamente mais tarde.' });
    }
  } catch (err) {
    console.error(err);

    return res.status(500).json({ success: false, message: 'Erro ao registrar usuário. Verifique os dados e tente novamente.' });
  }
});





app.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ success: false, message: 'Missing fields' });

    const response = await callIdentityToolkit('signInWithPassword', { email, password, returnSecureToken: true });
    return res.json({ success: true, data: response.data });
  } catch (err) {
    console.error(err.response ? err.response.data : err.message);
    const message = err.response && err.response.data ? err.response.data : { message: err.message };
    return res.status(400).json({ success: false, message });
  }
});


app.post('/resetPassword', async (req, res) => {
  try {
    const { email } = req.body;
    if (!email) return res.status(400).json({ success: false, message: 'Email obrigatório' });
    const response = await callIdentityToolkit('sendOobCode', { requestType: 'PASSWORD_RESET', email });
    return res.json({ success: true, data: response.data });
  } catch (err) {
    console.error(err.response ? err.response.data : err.message);
    return res.status(400).json({ success: false, message: err.response ? err.response.data : err.message });
  }
});


const requireAuth = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization || '';
    if (!authHeader.startsWith('Bearer ')) return res.status(401).json({ success: false, message: 'Unauthorized' });
    const idToken = authHeader.split(' ')[1];
    const decoded = await admin.auth().verifyIdToken(idToken);
    req.user = decoded;
    return next();
  } catch (err) {
    console.error('auth verify failed', err.message);
    return res.status(401).json({ success: false, message: 'Invalid token' });
  }
};


app.post('/updateProfile', requireAuth, async (req, res) => {
  try {
    const uid = req.user.uid;
    const { newEmail, newDisplayName } = req.body;
    const update = {};
    if (newEmail) update.email = newEmail;
    if (newDisplayName) update.displayName = newDisplayName;
    if (Object.keys(update).length === 0) return res.status(400).json({ success: false, message: 'No fields to update' });

    const userRecord = await admin.auth().updateUser(uid, update);

    await admin.firestore().collection('users').doc(uid).update({
      ...(newEmail ? { email: newEmail } : {}),
      ...(newDisplayName ? { displayName: newDisplayName } : {})
    });
    return res.json({ success: true, data: userRecord });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, message: err.message });
  }
});


app.post('/changePassword', requireAuth, async (req, res) => {
  try {
    const uid = req.user.uid;
    const { newPassword } = req.body;
    if (!newPassword || newPassword.length < 6) return res.status(400).json({ success: false, message: 'Senha inválida' });
    const userRecord = await admin.auth().updateUser(uid, { password: newPassword });
    return res.json({ success: true, data: userRecord });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, message: err.message });
  }
});


app.post('/deleteAccount', requireAuth, async (req, res) => {
  try {
    const uid = req.user.uid;

    await admin.auth().deleteUser(uid);

    await admin.firestore().collection('users').doc(uid).delete();
    return res.json({ success: true });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

