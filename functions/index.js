const functions = require('firebase-functions');
const admin = require('firebase-admin');
const express = require('express');
const cors = require('cors');
const axios = require('axios');
// SendGrid
const sgMail = require('@sendgrid/mail');

// Configure SendGrid API key from functions.config, .runtimeconfig.json or env
(() => {
  // Prefer environment variables (Secret Manager mapped) first. If not present and running locally, fall back to .runtimeconfig.json
  let sgKey = process.env.SENDGRID_API_KEY || process.env.SG_API_KEY || '';
  if (!sgKey) {
    try {
      const rc = require('./.runtimeconfig.json');
      if (rc && rc.sendgrid && rc.sendgrid.api_key) sgKey = rc.sendgrid.api_key;
    } catch (e) {
      // ignore
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

// Helper: try to determine if SendGrid is configured at runtime
const isSendGridConfigured = () => {
  if (process.env.SENDGRID_API_KEY || process.env.SG_API_KEY) return true;
  try {
    const rc = require('./.runtimeconfig.json');
    if (rc && rc.sendgrid && rc.sendgrid.api_key) return true;
  } catch (e) { /* ignore */ }
  return false;
  return false;
};

// Helper: produce Firestore Timestamp or fallback to number ms. Prefer admin.firestore.Timestamp when available.
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
    // ignore and fallback to numbers
  }
  return { createdAt: ms, expiresAt: expiryMs, _ms: ms };
};

// Helper: normalize a Firestore timestamp-like field to milliseconds
const millisFromFirestoreValue = (val) => {
  if (!val) return null;
  try {
    if (typeof val.toMillis === 'function') return val.toMillis();
  } catch (e) { /* ignore */ }
  if (typeof val === 'number') return val;
  // Firestore REST sometimes returns { _seconds, _nanoseconds }
  if (val && typeof val._seconds === 'number') return (val._seconds * 1000) + Math.floor((val._nanoseconds || 0) / 1e6);
  return null;
};

// Uso: POST /sendFeedback { user_email, user_name, subject, message }
app.post('/sendFeedback', async (req, res) => {
  try {
    const { user_email, user_name, subject, message } = req.body;
    if (!user_email || !user_name || !subject || !message) {
      return res.status(400).json({ success: false, message: 'Missing fields' });
    }

    // Prefer SendGrid for server-side sending. Determine feedback inbox from env/config.
    let feedback_email = process.env.FEEDBACK_EMAIL || process.env.SENDGRID_FROM || '';
    if (!feedback_email) {
      try { const rc = require('./.runtimeconfig.json'); if (rc && rc.sendgrid && rc.sendgrid.feedback_email) feedback_email = rc.sendgrid.feedback_email || ''; } catch (e) { /* ignore */ }
    }

    const isEmulator = !!process.env.FIRESTORE_EMULATOR_HOST || !!process.env.FIREBASE_EMULATOR_HUB;
    const hasSG = isSendGridConfigured();

    // Determine sender candidate. Prefer explicit env SENDGRID_FROM (mapped from Secret Manager), then functions.config, then runtimeconfig/emailjs fallback
    let sgFrom = process.env.SENDGRID_FROM || '';
    if (!sgFrom) {
      try { const rc = require('./.runtimeconfig.json'); if (rc && rc.sendgrid && rc.sendgrid.from) sgFrom = rc.sendgrid.from || ''; } catch (e) { /* ignore */ }
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
        // Ensure sgFrom is set
        if (!sgFrom && !isEmulator) {
          return res.status(500).json({ success: false, message: 'sendgrid.from não está configurado. Defina um remetente verificado.' });
        }
        // Force 'from' to the verified sender and 'to' to the feedback inbox (feedback_email)
        msg.from = sgFrom || msg.from;
        msg.to = feedback_email || msg.to;
        await sgMail.send(msg);
        return res.json({ success: true, via: 'sendgrid' });
      }

      // Emulator fallback: simulate success so dev flow continues
      console.warn('SendGrid não configurado; modo emulador - simulando envio de feedback');
      console.log('Simulated feedback message:', msg);
      return res.json({ success: true, simulated: true });
    } catch (err) {
      // Log richer error info to help debug SendGrid responses
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

// --- Novos endpoints para autenticação e perfil ---

// Helper: chama a REST API do Identity Toolkit (Firebase Auth REST)
const callIdentityToolkit = async (path, payload) => {
  // Tenta obter apiKey de functions.config(); se não disponível, tenta carregar .runtimeconfig.json ou variável de ambiente
  let apiKey = '';
  // Prefer environment vars first
  apiKey = process.env.FIREBASE_API_KEY || process.env.FIREBASE_WEB_API_KEY || '';
  if (!apiKey) {
    try {
      // carregar arquivo local usado pelo emulator
      // eslint-disable-next-line global-require, import/no-dynamic-require
      const rc = require('./.runtimeconfig.json');
      if (rc && rc.firebase && rc.firebase.api_key) apiKey = rc.firebase.api_key;
    } catch (e) {
      // ignore if file missing
    }
  }

  if (!apiKey) throw new Error('Firebase API key not configured in functions.config().firebase.api_key or .runtimeconfig.json or env FIREBASE_API_KEY');
  const url = `https://identitytoolkit.googleapis.com/v1/accounts:${path}?key=${apiKey}`;
  return axios.post(url, payload, { headers: { 'Content-Type': 'application/json' } });
};

// Registro: cria usuário via Admin SDK e retorna um customToken para login no cliente
app.post('/register', async (req, res) => {
  try {
    const { email, password, displayName, accepted_terms_of_service, accepted_privacy_policy } = req.body || {};
    if (!email || !password || !displayName) return res.status(400).json({ success: false, message: 'Missing fields' });

    // validações simples
    if (displayName.trim().length < 3) return res.status(400).json({ success: false, message: 'Nome muito curto' });
    if (password.length < 6) return res.status(400).json({ success: false, message: 'Senha deve ter ao menos 6 caracteres' });

    // cria usuário
    let userRecord;
    try {
      userRecord = await admin.auth().createUser({ email, password, displayName });
    } catch (e) {
      // detecta email já existente e retorna mensagem amigável
      if (e && e.code === 'auth/email-already-exists') {
        return res.status(400).json({ success: false, message: 'Usuário já existe. Se você não lembra a senha, use o campo "Esqueci minha senha".' });
      }
      throw e;
    }

    // opcional: criar documento de perfil no Firestore
    // createdAt: tenta usar serverTimestamp, senão usa Date.now()
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

    // Se o cliente informou aceitação de termos/privacy, salve server-side para evitar problemas com Firestore REST auth
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
      // Não bloquear o registro por falha aqui; cliente pode tentar novamente
    }

    // cria custom token para o cliente autenticar com signInWithCustomToken
    try {
      const customToken = await admin.auth().createCustomToken(userRecord.uid);
      // Retornamos o customToken e o uid. O cliente pode usar signInWithCustomToken(customToken)
      return res.json({ success: true, data: { uid: userRecord.uid, customToken }, message: 'Usuário criado com sucesso. Use o token para efetuar login.' });
    } catch (err) {
      console.error('Error creating custom token', err && err.message ? err.message : err);
      return res.status(500).json({ success: false, message: 'Erro interno ao criar credenciais. Tente novamente mais tarde.' });
    }
  } catch (err) {
    console.error(err);
    // Retorne mensagem amigável
    return res.status(500).json({ success: false, message: 'Erro ao registrar usuário. Verifique os dados e tente novamente.' });
  }
});

// Email verification endpoints were removed to simplify registration flow.
// Token-based email verification was adding complexity to the client; if you need it later we can reintroduce a simpler flow.

// Login: recebe email+password, usa Identity Toolkit REST para gerar idToken/refresh
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

// Reset de senha: envia email de reset via REST API
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

// Middleware: verifica idToken enviado no header Authorization: Bearer <idToken>
const requireAuth = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization || '';
    if (!authHeader.startsWith('Bearer ')) return res.status(401).json({ success: false, message: 'Unauthorized' });
    const idToken = authHeader.split(' ')[1];
    const decoded = await admin.auth().verifyIdToken(idToken);
    req.user = decoded; // contém uid, email, etc.
    return next();
  } catch (err) {
    console.error('auth verify failed', err.message);
    return res.status(401).json({ success: false, message: 'Invalid token' });
  }
};

// Atualizar perfil (nome e/ou email) - protege com idToken
app.post('/updateProfile', requireAuth, async (req, res) => {
  try {
    const uid = req.user.uid;
    const { newEmail, newDisplayName } = req.body;
    const update = {};
    if (newEmail) update.email = newEmail;
    if (newDisplayName) update.displayName = newDisplayName;
    if (Object.keys(update).length === 0) return res.status(400).json({ success: false, message: 'No fields to update' });

    const userRecord = await admin.auth().updateUser(uid, update);
    // atualizar perfil no Firestore
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

// Change password (requires auth)
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

// Deletar conta (requires auth)
app.post('/deleteAccount', requireAuth, async (req, res) => {
  try {
    const uid = req.user.uid;
    // deletar user
    await admin.auth().deleteUser(uid);
    // remover doc do Firestore
    await admin.firestore().collection('users').doc(uid).delete();
    return res.json({ success: true });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

