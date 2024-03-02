
  // Import the functions you need from the SDKs you need
  import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-app.js";
  import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-analytics.js";
  import { getAuth, signInWithCustomToken } from "firebase/auth";
  // TODO: Add SDKs for Firebase products that you want to use
  // https://firebase.google.com/docs/web/setup#available-libraries

  // Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  const firebaseConfig = {
    apiKey: "AIzaSyBwHdFjIA2J9tzljYInQdJtaeviLzFrCZ4",
    authDomain: "stockers-2f8d2.firebaseapp.com",
    projectId: "stockers-2f8d2",
    storageBucket: "stockers-2f8d2.appspot.com",
    messagingSenderId: "428351217361",
    appId: "1:428351217361:web:52e8660bd1ecf27d192ff0",
    measurementId: "G-ZBCBHCVEQM"
  };

  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);
  console.log(app);