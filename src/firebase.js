// Import the functions you need from the SDKs you need

import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyD7FWeJ1ok2zbuRLtA_TNjeHvNBP5j--Gs",
  authDomain: "react-auth-tutorial-f6e47.firebaseapp.com",
  projectId: "react-auth-tutorial-f6e47",
  storageBucket: "react-auth-tutorial-f6e47.appspot.com",
  messagingSenderId: "175797028406",
  appId: "1:175797028406:web:d7ed18f6819ef150f87a97"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// Initialize Firebase Authentication and get a reference to the service
const auth = getAuth(app);
export { auth };